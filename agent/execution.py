"""SQL execution helper (provided complete).

execute_sql() runs the agent's SQL against the target DB in read-only mode
and returns a structured ExecutionResult. The verify node consumes this
to decide whether the answer looks plausible.
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass

from agent.schema import db_path


@dataclass
class ExecutionResult:
    ok: bool
    rows: list[tuple] | None = None
    columns: list[str] | None = None
    error: str | None = None
    row_count: int = 0

    def render(
        self,
        max_rows: int = 5,
        max_cell_chars: int = 300,
        max_preview_chars: int = 4000,
    ) -> str:
        """Compact text rendering for prompt context."""
        if not self.ok:
            return f"ERROR: {self.error}"
        if self.row_count == 0:
            return "OK: 0 rows returned."
        cols = ", ".join(self.columns or [])

        def fmt_cell(value) -> str:
            text = "" if value is None else str(value)
            text = text.replace("\n", "\\n")
            if len(text) > max_cell_chars:
                return text[:max_cell_chars] + "...[truncated]"
            return text

        preview_lines: list[str] = []
        used_chars = 0
        for row in (self.rows or [])[:max_rows]:
            line = " | ".join(fmt_cell(c) for c in row)
            if used_chars + len(line) > max_preview_chars:
                remaining = max_preview_chars - used_chars
                if remaining > 0:
                    preview_lines.append(line[:remaining] + "...[preview truncated]")
                break
            preview_lines.append(line)
            used_chars += len(line)

        preview = "\n".join(preview_lines)
        more = f"\n... ({self.row_count - max_rows} more rows)" if self.row_count > max_rows else ""
        return f"OK: {self.row_count} rows.\nCOLUMNS: {cols}\nFIRST ROWS:\n{preview}{more}"


def execute_sql(db_id: str, sql: str, timeout_seconds: float = 5.0) -> ExecutionResult:
    """Run SQL against db_id's sqlite, return result or error."""
    path = db_path(db_id)
    try:
        with sqlite3.connect(
            f"file:{path}?mode=ro",
            uri=True,
            timeout=timeout_seconds,
        ) as conn:
            cur = conn.execute(sql)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = cur.fetchall()
            return ExecutionResult(ok=True, rows=rows, columns=cols, row_count=len(rows))
    except Exception as e:  # noqa: BLE001
        return ExecutionResult(ok=False, error=f"{type(e).__name__}: {e}")
