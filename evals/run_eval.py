"""Eval runner using execution accuracy.

Reads evals/eval_set.jsonl, calls the agent at AGENT_URL on each question,
then compares the agent's SQL output to the gold SQL by *executed rows*
(canonicalized: sorted, stringified, None-coerced to empty).

Helpers (run_sql / canonicalize / matches) are provided. You implement
eval_one() and summarize().

Run:
    uv run python evals/run_eval.py --out results/eval_baseline.json
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EVAL_FILE = ROOT / "evals" / "eval_set.jsonl"
DEFAULT_OUT_FILE = ROOT / "results" / "eval_baseline.json"
DB_DIR = ROOT / "data" / "bird"
AGENT_URL_DEFAULT = "http://localhost:8001/answer"
REPORT_ITERATIONS = 3


# ---------- Helpers (provided) -----------------------------------------

def run_sql(db_id: str, sql: str, timeout: float = 5.0) -> tuple[bool, list[tuple] | None, str | None]:
    """Run sql against db_id in read-only mode. Returns (ok, rows, error)."""
    path = DB_DIR / f"{db_id}.sqlite"
    try:
        with sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=timeout) as conn:
            cur = conn.execute(sql)
            rows = cur.fetchall()
            return True, rows, None
    except Exception as e:  # noqa: BLE001
        return False, None, f"{type(e).__name__}: {e}"


def canonicalize(rows: list[tuple] | None) -> list[tuple] | None:
    """Sort rows; coerce cells to str; None -> ''."""
    if rows is None:
        return None
    return sorted(tuple("" if c is None else str(c) for c in row) for row in rows)


def matches(gold_rows: list[tuple] | None, pred_rows: list[tuple] | None) -> bool:
    if gold_rows is None or pred_rows is None:
        return False
    return canonicalize(gold_rows) == canonicalize(pred_rows)


# ---------- Implement these (Phase 5) ----------------------------------

def eval_one(question: dict, agent_url: str) -> dict:
    """Score one question. Return a dict capturing per-iteration correctness."""
    db_id = question["db_id"]
    gold_sql = question["gold_sql"]

    gold_ok, gold_rows, gold_error = run_sql(db_id, gold_sql)

    agent_payload = {
        "question": question["question"],
        "db": db_id,
        "tags": {
            "phase": "eval",
            "db_id": db_id,
        },
    }

    t0 = time.monotonic()
    agent_json: dict | None = None
    agent_error: str | None = None
    http_status: int | None = None
    try:
        with httpx.Client(timeout=180.0) as client:
            resp = client.post(agent_url, json=agent_payload)
        http_status = resp.status_code
        if resp.status_code == 200:
            agent_json = resp.json()
        else:
            agent_error = resp.text
    except Exception as e:  # noqa: BLE001
        agent_error = f"{type(e).__name__}: {e}"
    latency = time.monotonic() - t0

    pred_sql = (agent_json or {}).get("sql", "")
    pred_ok = False
    pred_rows = None
    pred_error = "agent did not return SQL"
    if pred_sql:
        pred_ok, pred_rows, pred_error = run_sql(db_id, pred_sql)
    final_correct = gold_ok and pred_ok and matches(gold_rows, pred_rows)

    attempts: list[dict] = []
    seen_attempts: set[tuple[int, str]] = set()
    for item in (agent_json or {}).get("history", []):
        if item.get("node") not in {"generate_sql", "revise"}:
            continue
        sql = item.get("sql", "")
        iteration = int(item.get("iteration") or 0)
        if not sql or iteration <= 0:
            continue
        key = (iteration, sql)
        if key in seen_attempts:
            continue
        seen_attempts.add(key)

        attempt_ok, attempt_rows, attempt_error = run_sql(db_id, sql)
        attempts.append({
            "iteration": iteration,
            "node": item.get("node"),
            "sql": sql,
            "execution_ok": attempt_ok,
            "execution_error": attempt_error,
            "correct": gold_ok and attempt_ok and matches(gold_rows, attempt_rows),
        })

    final_iteration = int((agent_json or {}).get("iterations") or 0)
    if pred_sql and final_iteration > 0 and (final_iteration, pred_sql) not in seen_attempts:
        attempts.append({
            "iteration": final_iteration,
            "node": "final",
            "sql": pred_sql,
            "execution_ok": pred_ok,
            "execution_error": pred_error,
            "correct": final_correct,
        })

    attempts.sort(key=lambda a: a["iteration"])

    return {
        "question": question["question"],
        "db_id": db_id,
        "gold_sql": gold_sql,
        "gold_execution_ok": gold_ok,
        "gold_execution_error": gold_error,
        "agent_http_status": http_status,
        "agent_error": agent_error or (agent_json or {}).get("error"),
        "agent_latency_seconds": latency,
        "agent_iterations": final_iteration,
        "agent_ok": bool((agent_json or {}).get("ok", False)),
        "pred_sql": pred_sql,
        "pred_execution_ok": pred_ok,
        "pred_execution_error": pred_error,
        "correct": final_correct,
        "attempts": attempts,
        "history": (agent_json or {}).get("history", []),
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate per-question results.

    Per-iteration carry-forward: if the agent terminated at iteration j < k
    (verify said ok at j, or it hit MAX_ITERATIONS at j < k), treat the
    question's iteration-k result as identical to its iteration-j result.
    The agent stopped emitting; whatever it had at termination is what
    would have been served had we polled at iteration k.
    """
    total = len(results)
    if total == 0:
        return {
            "total": 0,
            "correct": 0,
            "execution_accuracy": 0.0,
            "per_iteration": {},
        }

    correct = sum(1 for r in results if r.get("correct"))
    agent_ok = sum(1 for r in results if r.get("agent_ok"))
    agent_errors = sum(1 for r in results if r.get("agent_error"))
    gold_errors = sum(1 for r in results if not r.get("gold_execution_ok"))
    pred_execution_errors = sum(1 for r in results if not r.get("pred_execution_ok"))
    max_observed_iteration = max(
        [REPORT_ITERATIONS]
        + [int(r.get("agent_iterations") or 0) for r in results]
        + [int(a.get("iteration") or 0) for r in results for a in r.get("attempts", [])]
    )

    per_iteration: dict[str, dict] = {}
    for iteration in range(1, max_observed_iteration + 1):
        iteration_correct = 0
        answered = 0
        for r in results:
            attempts = [
                a for a in r.get("attempts", [])
                if int(a.get("iteration") or 0) <= iteration
            ]
            if not attempts:
                continue
            answered += 1
            if attempts[-1].get("correct"):
                iteration_correct += 1
        per_iteration[str(iteration)] = {
            "correct": iteration_correct,
            "answered": answered,
            "total": total,
            "pass_rate": iteration_correct / total,
        }

    latencies = sorted(float(r.get("agent_latency_seconds") or 0.0) for r in results)

    def pct(p: float) -> float:
        if not latencies:
            return 0.0
        k = int(round(p * (len(latencies) - 1)))
        return latencies[k]

    return {
        "total": total,
        "correct": correct,
        "execution_accuracy": correct / total,
        "agent_ok": agent_ok,
        "agent_error_count": agent_errors,
        "gold_execution_error_count": gold_errors,
        "pred_execution_error_count": pred_execution_errors,
        "latency_seconds": {
            "p50": pct(0.50),
            "p95": pct(0.95),
            "max": latencies[-1] if latencies else 0.0,
        },
        "per_iteration": per_iteration,
    }


# ---------- Main (provided) --------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-set", type=Path, default=DEFAULT_EVAL_FILE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_FILE)
    parser.add_argument("--agent-url", default=AGENT_URL_DEFAULT)
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    args = parser.parse_args()

    questions = [json.loads(line) for line in args.eval_set.read_text().splitlines() if line.strip()]
    if args.limit is not None:
        questions = questions[:args.limit]
    print(f"Loaded {len(questions)} eval questions from {args.eval_set}")

    results: list[dict] = []
    t0 = time.monotonic()
    for i, q in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {q['db_id']}: {q['question'][:60]}...", flush=True)
        results.append(eval_one(q, args.agent_url))
    elapsed = time.monotonic() - t0

    summary = summarize(results)
    out = {
        "summary": summary,
        "wall_clock_seconds": elapsed,
        "results": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
