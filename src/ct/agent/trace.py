"""Compatibility trace logger used by CLI diagnostics and exports."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any


class TraceLogger:
    """Small JSONL trace helper with enough API for CLI tooling."""

    def __init__(self, session_id: str, events: list[dict[str, Any]] | None = None):
        self.session_id = session_id
        self.events: list[dict[str, Any]] = list(events or [])

    @staticmethod
    def traces_dir() -> Path:
        """Use the shared sessions directory where trace JSONL files already live."""

        directory = Path.home() / ".ct" / "sessions"
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    def _append(self, event_type: str, **payload: Any) -> None:
        payload.setdefault("timestamp", time.time())
        payload["type"] = event_type
        self.events.append(payload)

    def query_start(self, query: str, model: str = "") -> None:
        self._append(
            "query_start",
            session_id=self.session_id,
            query=query,
            model=model,
        )

    def plan(self, steps: list[Any], query: str = "") -> None:
        self._append("plan", query=query, steps=steps)

    def step_start(self, step_id: int, tool: str, tool_args: dict[str, Any] | None = None) -> None:
        self._append("step_start", step_id=step_id, tool=tool, tool_args=tool_args or {})

    def step_complete(
        self,
        step_id: int,
        tool: str,
        result: dict[str, Any] | None = None,
        duration_s: float = 0.0,
    ) -> None:
        self._append(
            "step_complete",
            step_id=step_id,
            tool=tool,
            result=result or {},
            duration_s=duration_s,
        )

    def step_fail(
        self,
        step_id: int,
        tool: str,
        error: str,
        duration_s: float = 0.0,
    ) -> None:
        self._append(
            "step_fail",
            step_id=step_id,
            tool=tool,
            error=error,
            duration_s=duration_s,
        )

    def step_retry(self, step_id: int, tool: str, reason: str = "") -> None:
        self._append("step_retry", step_id=step_id, tool=tool, reason=reason)

    def synthesize_start(self) -> None:
        self._append("synthesize_start")

    def synthesize_end(self, token_count: int = 0, duration_s: float = 0.0) -> None:
        self._append(
            "synthesize_end",
            token_count=token_count,
            duration_s=duration_s,
        )

    def query_end(
        self,
        iterations: int = 1,
        total_steps: int = 0,
        completed_steps: int = 0,
        failed_steps: int = 0,
        duration_s: float = 0.0,
        cost_usd: float = 0.0,
    ) -> None:
        self._append(
            "query_end",
            iterations=iterations,
            total_steps=total_steps,
            completed_steps=completed_steps,
            failed_steps=failed_steps,
            duration_s=duration_s,
            cost_usd=cost_usd,
        )

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else self.traces_dir() / f"{self.session_id}.trace.jsonl"
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as handle:
            for event in self.events:
                handle.write(json.dumps(event, default=str) + "\n")
        return target

    @classmethod
    def load(cls, path: Path | str) -> "TraceLogger":
        target = Path(path)
        events: list[dict[str, Any]] = []
        with open(target, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))

        session_id = target.stem.replace(".trace", "")
        for event in events:
            if event.get("type") == "query_start" and event.get("session_id"):
                session_id = str(event["session_id"])
                break

        return cls(session_id=session_id, events=events)

    def diagnostics(self) -> dict[str, Any]:
        queries: list[dict[str, Any]] = []
        current: dict[str, Any] | None = None

        for event in self.events:
            event_type = event.get("type", "")

            if event_type == "query_start":
                current = {
                    "query_number": len(queries) + 1,
                    "query": event.get("query", ""),
                    "closed": False,
                    "plan_count": 0,
                    "activity_count": 0,
                    "step_start_count": 0,
                    "step_complete_count": 0,
                    "step_fail_count": 0,
                    "step_retry_count": 0,
                    "synthesize_start_count": 0,
                    "synthesize_end_count": 0,
                }
                queries.append(current)
                continue

            if current is None:
                continue

            if event_type == "text":
                content = event.get("content", "")
                if isinstance(content, str) and content.strip():
                    current["activity_count"] += 1
            elif event_type == "plan":
                current["plan_count"] += 1
                current["activity_count"] += 1
            elif event_type in {"step_start", "tool_start"}:
                current["step_start_count"] += 1
                current["activity_count"] += 1
            elif event_type == "step_complete":
                current["step_complete_count"] += 1
                current["activity_count"] += 1
            elif event_type == "tool_result":
                if event.get("is_error"):
                    current["step_fail_count"] += 1
                else:
                    current["step_complete_count"] += 1
                current["activity_count"] += 1
            elif event_type == "step_fail":
                current["step_fail_count"] += 1
                current["activity_count"] += 1
            elif event_type == "step_retry":
                current["step_retry_count"] += 1
                current["activity_count"] += 1
            elif event_type == "synthesize_start":
                current["synthesize_start_count"] += 1
                current["activity_count"] += 1
            elif event_type == "synthesize_end":
                current["synthesize_end_count"] += 1
                current["activity_count"] += 1
            elif event_type == "query_end":
                current["closed"] = True

        query_start_count = sum(1 for event in self.events if event.get("type") == "query_start")
        query_end_count = sum(1 for event in self.events if event.get("type") == "query_end")

        unclosed_queries = [
            query["query_number"] for query in queries if not query["closed"]
        ]
        queries_with_failures = [
            query["query_number"] for query in queries if query["step_fail_count"] > 0
        ]
        queries_with_no_plan = [
            query["query_number"]
            for query in queries
            if query["plan_count"] == 0 and query["activity_count"] == 0
        ]
        queries_with_no_completion = [
            query["query_number"]
            for query in queries
            if not query["closed"] or query["activity_count"] == 0
        ]
        queries_with_synthesis_mismatch = [
            query["query_number"]
            for query in queries
            if query["synthesize_start_count"] != query["synthesize_end_count"]
        ]

        return {
            "session_id": self.session_id,
            "event_count": len(self.events),
            "query_count": len(queries),
            "query_start_count": query_start_count,
            "query_end_count": query_end_count,
            "total_step_start_count": sum(query["step_start_count"] for query in queries),
            "total_step_complete_count": sum(
                query["step_complete_count"] for query in queries
            ),
            "total_step_fail_count": sum(query["step_fail_count"] for query in queries),
            "total_step_retry_count": sum(query["step_retry_count"] for query in queries),
            "unclosed_queries": unclosed_queries,
            "queries_with_failures": queries_with_failures,
            "queries_with_no_plan": queries_with_no_plan,
            "queries_with_no_completion": queries_with_no_completion,
            "queries_with_synthesis_mismatch": queries_with_synthesis_mismatch,
            "queries": queries,
        }

    def query_summaries(self) -> list[dict[str, Any]]:
        summaries = []
        for query in self.diagnostics()["queries"]:
            summaries.append(
                {
                    "query_number": query["query_number"],
                    "query": query["query"],
                    "closed": query["closed"],
                    "step_start_count": query["step_start_count"],
                    "step_complete_count": query["step_complete_count"],
                    "step_fail_count": query["step_fail_count"],
                    "step_retry_count": query["step_retry_count"],
                }
            )
        return summaries

    def to_text(self) -> str:
        lines = []
        for event in self.events:
            event_type = event.get("type", "")
            if event_type == "query_start":
                lines.append(f"query_start: {event.get('query', '')}")
            elif event_type == "plan":
                lines.append("plan")
            elif event_type in {"step_start", "tool_start"}:
                lines.append(f"{event_type}: {event.get('tool', '')}")
            elif event_type in {"step_complete", "tool_result"}:
                lines.append(f"{event_type}: {event.get('tool', '')}")
            elif event_type == "query_end":
                lines.append("query_end")
            else:
                lines.append(event_type)
        return "\n".join(lines)
