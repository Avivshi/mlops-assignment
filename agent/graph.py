"""LangGraph agent: text-to-SQL with verify+revise loop.

Graph shape:

    START -> attach_schema -> generate_sql -> execute -> verify
                                                          |
                                              ok=true ----+----> END
                                                          |
                                              ok=false ---+----> revise -> execute -> verify (loop)

Loop is capped at MAX_ITERATIONS total generate/revise calls.

The execute node and the graph wiring are provided. `generate_sql_node` is
filled in as a worked example; you implement `verify`, `revise`, and the
conditional router following the same shape.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from agent import prompts
from agent.execution import ExecutionResult, execute_sql
from agent.schema import render_schema

# Total generate + revise calls before the loop is forced to stop.
# 3-5 is a reasonable range; tune it as part of Phase 3.
MAX_ITERATIONS = 3

DEFAULT_VLLM_BASE_URL = "http://localhost:8000/v1"
DEFAULT_VLLM_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"

VLLM_BASE_URL = os.environ.get("VLLM_BASE_URL", DEFAULT_VLLM_BASE_URL)
VLLM_MODEL = os.environ.get("VLLM_MODEL", DEFAULT_VLLM_MODEL)

# vLLM ignores the key, but a hosted OpenAI-compatible provider needs a real one.
# Lets you point the agent at e.g. OpenAI while iterating without a running vLLM.
LLM_API_KEY = os.environ.get("OPENAI_API_KEY", "not-needed")


class VerifyDecision(BaseModel):
    ok: bool = Field(description="Whether the SQL result correctly answers the user question.")
    issue: str = Field(description="Short explanation when ok is false; empty string when ok is true.")


@dataclass
class AgentState:
    """State threaded through the graph. Extend with fields you need."""

    question: str
    db_id: str
    schema: str = ""
    sql: str = ""
    execution: ExecutionResult | None = None
    verify_ok: bool = False
    verify_issue: str = ""
    iteration: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)


def llm() -> ChatOpenAI:
    """Chat client pointed at VLLM_BASE_URL (your local vLLM by default)."""
    return ChatOpenAI(
        model=VLLM_MODEL,
        base_url=VLLM_BASE_URL,
        api_key=LLM_API_KEY,
        temperature=0.0,
    )

# ---- Nodes ------------------------------------------------------------

def _attach_schema(state: AgentState) -> dict:
    """Provided. Render the DB schema once at the start of the run."""
    return {"schema": render_schema(state.db_id)}


def _extract_sql(text: str) -> str:
    """Pull a SQL statement out of an LLM reply, stripping markdown fences/prose.

    Intentionally simple: take the first ```sql ... ``` block if there is one,
    otherwise the whole reply. You may need to harden this for your prompts.
    """
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    sql = (fenced.group(1) if fenced else text).strip()
    first_keyword = re.search(r"\b(WITH|SELECT)\b", sql, re.IGNORECASE)
    if first_keyword:
        sql = sql[first_keyword.start() :]
    semicolon = sql.find(";")
    if semicolon != -1:
        sql = sql[: semicolon + 1]
    return sql.rstrip("`").strip()


def _message_text(content: Any) -> str:
    """Normalize LangChain message content to text."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    return str(content)


def _invoke_verifier(messages: list[tuple[str, str]]) -> VerifyDecision:
    """Call the verifier with schema-constrained JSON output."""
    try:
        return llm().with_structured_output(
            VerifyDecision,
            method="json_schema",
            strict=True,
        ).invoke(messages)
    except Exception as e:  # noqa: BLE001
        return VerifyDecision(
            ok=False,
            issue=f"Verifier failed to produce structured JSON: {type(e).__name__}: {e}",
        )


def _duplicate_row_issue(state: AgentState) -> str | None:
    """Catch exact duplicate result rows before spending an LLM verify call."""
    if state.execution is None or not state.execution.ok or not state.execution.rows:
        return None
    if state.execution.row_count <= 1:
        return None
    if re.search(r"\b(COUNT|SUM|AVG|MIN|MAX|GROUP\s+BY)\b", state.sql, re.IGNORECASE):
        return None

    rows = [tuple(row) for row in state.execution.rows]
    if len(rows) == len(set(rows)):
        return None
    return "Execution returned duplicate identical rows; use DISTINCT unless duplicates are explicitly requested."


def generate_sql_node(state: AgentState) -> dict:
    """Worked example - the other LLM nodes follow this same shape.

    Build messages from the prompts, call the shared llm(), extract the SQL,
    and return only the state fields you changed. `iteration` is bumped here
    (and in revise) so route_after_verify can enforce MAX_ITERATIONS.

    This node is wired and ready; fill in GENERATE_SQL_SYSTEM / GENERATE_SQL_USER
    in prompts.py to make it produce real queries.
    """
    response = llm().invoke([
        ("system", prompts.GENERATE_SQL_SYSTEM),
        ("user", prompts.GENERATE_SQL_USER.format(
            schema=state.schema,
            question=state.question,
        )),
    ])
    sql = _extract_sql(_message_text(response.content))
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "history": state.history + [{
            "node": "generate_sql",
            "iteration": state.iteration + 1,
            "sql": sql,
        }],
    }


def execute_node(state: AgentState) -> dict:
    """Provided. Runs the SQL and stores the result."""
    return {"execution": execute_sql(state.db_id, state.sql)}


def verify_node(state: AgentState) -> dict:
    """Decide whether state.execution plausibly answers state.question.

    Follow the generate_sql_node pattern, but ask the model for a structured
    VerifyDecision so vLLM can enforce the JSON schema through xgrammar.
    state.execution.render() gives you a compact view of the rows or error to
    feed into the prompt.

    Return: {"verify_ok": <bool>, "verify_issue": <str>}.
    What counts as "not plausible" is yours to define - see the Phase 3 targets
    in the README.
    """
    duplicate_issue = _duplicate_row_issue(state)
    if duplicate_issue is not None:
        return {
            "verify_ok": False,
            "verify_issue": duplicate_issue,
            "history": state.history + [{
                "node": "verify",
                "iteration": state.iteration,
                "ok": False,
                "issue": duplicate_issue,
            }],
        }

    execution_text = state.execution.render() if state.execution is not None else "No execution result."
    decision = _invoke_verifier([
        ("system", prompts.VERIFY_SYSTEM),
        ("user", prompts.VERIFY_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            execution=execution_text,
        )),
    ])
    return {
        "verify_ok": decision.ok,
        "verify_issue": decision.issue,
        "history": state.history + [{
            "node": "verify",
            "iteration": state.iteration,
            "ok": decision.ok,
            "issue": decision.issue,
        }],
    }


def revise_node(state: AgentState) -> dict:
    """Produce a revised SQL query given state.verify_issue and the prior attempt.

    Same shape as generate_sql_node, but the prompt should include the failing
    SQL, its execution result, and the verifier's complaint so the model can fix
    it. Bump the iteration counter the same way generate_sql_node does so the
    loop terminates.

    Return: {"sql": <str>, "iteration": state.iteration + 1, ...}.
    """
    execution_text = state.execution.render() if state.execution is not None else "No execution result."
    response = llm().invoke([
        ("system", prompts.REVISE_SYSTEM),
        ("user", prompts.REVISE_USER.format(
            schema=state.schema,
            question=state.question,
            sql=state.sql,
            execution=execution_text,
            issue=state.verify_issue,
        )),
    ])
    sql = _extract_sql(_message_text(response.content))
    return {
        "sql": sql,
        "iteration": state.iteration + 1,
        "verify_ok": False,
        "verify_issue": "",
        "history": state.history + [{
            "node": "revise",
            "iteration": state.iteration + 1,
            "sql": sql,
            "previous_issue": state.verify_issue,
        }],
    }


def route_after_verify(state: AgentState) -> str:
    """Conditional router: return "revise" to loop, "end" to terminate.

    Two reasons to end: the verifier was happy (state.verify_ok), or you've hit
    the iteration cap (state.iteration >= MAX_ITERATIONS). Otherwise, revise.
    """
    if state.verify_ok:
        return "end"
    if state.iteration >= MAX_ITERATIONS:
        return "end"
    return "revise"


# ---- Graph wiring -----------------------------------------------------

def build_graph():
    g = StateGraph(AgentState)
    g.add_node("attach_schema", _attach_schema)
    g.add_node("generate_sql", generate_sql_node)
    g.add_node("execute", execute_node)
    g.add_node("verify", verify_node)
    g.add_node("revise", revise_node)

    g.add_edge(START, "attach_schema")
    g.add_edge("attach_schema", "generate_sql")
    g.add_edge("generate_sql", "execute")
    g.add_edge("execute", "verify")
    g.add_conditional_edges(
        "verify",
        route_after_verify,
        {"revise": "revise", "end": END},
    )
    g.add_edge("revise", "execute")
    return g.compile()


graph = build_graph()
