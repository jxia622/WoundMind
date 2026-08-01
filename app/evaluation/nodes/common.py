from __future__ import annotations

from typing import Any

from app.evaluation.schemas import AuditEvent


def append_audit(
    state: dict[str, Any],
    *,
    node: str,
    route: str | None = None,
    question: str | None = None,
    details: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    events = list(state.get("audit_log", []))
    events.append(
        AuditEvent(
            sequence=len(events) + 1,
            node=node,
            iteration=int(state.get("iteration_count", 0)),
            route=route,
            question=question,
            details=details or {},
        ).model_dump()
    )
    return events
