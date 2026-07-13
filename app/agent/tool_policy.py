from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any


POLICY_PATH = Path(__file__).with_name("condition_policy_registry.json")


def normalize_policy_key(value: str | None) -> str:
    if not value:
        return ""
    normalized = value.lower().replace("&", "and")
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    return re.sub(r"_+", "_", normalized).strip("_")


@lru_cache(maxsize=4)
def load_condition_policy_registry(path: str | None = None) -> dict[str, Any]:
    registry_path = Path(path) if path else POLICY_PATH
    return json.loads(registry_path.read_text())


def get_condition_policy(
    condition: str | None,
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = registry or load_condition_policy_registry()
    condition_key = normalize_policy_key(condition)

    for policy in registry.get("conditions", []):
        candidates = [
            policy.get("condition_id"),
            policy.get("condition_name"),
            *policy.get("classifier_labels", []),
            *policy.get("aliases", []),
        ]
        if condition_key in {normalize_policy_key(candidate) for candidate in candidates}:
            return policy

    return {
        "condition_id": "unknown",
        "condition_name": condition or "Unknown",
        "classifier_labels": [condition] if condition else [],
        "aliases": [],
        "primary_severity_axes": [],
        "network_features": [],
        "required_tools": [],
        "optional_tools": [],
        "clinical_doc_ids": [],
        "retrieval_filters": {"condition": "unknown"},
        "output_schema": "unsupported_condition_schema",
        "limitations": ["No condition policy is configured for this classifier output."],
    }


def build_tool_plan(
    condition_policy: dict[str, Any],
    registry: dict[str, Any] | None = None,
) -> dict[str, Any]:
    registry = registry or load_condition_policy_registry()
    tool_registry = registry.get("tools", {})

    runnable_tools: list[dict[str, Any]] = []
    missing_required_tools: list[dict[str, Any]] = []
    missing_optional_tools: list[dict[str, Any]] = []
    available_optional_tools: list[dict[str, Any]] = []

    for tool_id in condition_policy.get("required_tools", []):
        tool = _tool_metadata(tool_id, tool_registry)
        if _is_runtime_tool(tool):
            runnable_tools.append(_runtime_tool_entry(tool_id, tool, "required"))
        else:
            missing_required_tools.append(_missing_tool_entry(tool_id, tool))

    for tool_id in condition_policy.get("optional_tools", []):
        tool = _tool_metadata(tool_id, tool_registry)
        if _is_runtime_tool(tool):
            available_optional_tools.append(_runtime_tool_entry(tool_id, tool, "optional"))
        else:
            missing_optional_tools.append(_missing_tool_entry(tool_id, tool))

    return {
        "condition_id": condition_policy.get("condition_id"),
        "condition_name": condition_policy.get("condition_name"),
        "severity_axes": condition_policy.get("primary_severity_axes", []),
        "network_features": condition_policy.get("network_features", []),
        "runnable_tools": runnable_tools,
        "available_optional_tools": available_optional_tools,
        "missing_required_tools": missing_required_tools,
        "missing_optional_tools": missing_optional_tools,
        "clinical_doc_ids": condition_policy.get("clinical_doc_ids", []),
        "output_schema": condition_policy.get("output_schema"),
        "limitations": condition_policy.get("limitations", []),
    }


def runtime_tool_names(tool_plan: dict[str, Any] | None) -> set[str]:
    if not tool_plan:
        return set()
    return {
        tool["runtime_tool"]
        for tool in tool_plan.get("runnable_tools", [])
        if tool.get("runtime_tool")
    }


def summarize_tool_plan(tool_plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition_id": tool_plan.get("condition_id"),
        "severity_axes": tool_plan.get("severity_axes", []),
        "network_features": tool_plan.get("network_features", []),
        "runnable_tools": [
            {
                "tool_id": tool.get("tool_id"),
                "runtime_tool": tool.get("runtime_tool"),
                "source": tool.get("source"),
            }
            for tool in tool_plan.get("runnable_tools", [])
        ],
        "available_optional_tools": [
            tool.get("tool_id") for tool in tool_plan.get("available_optional_tools", [])
        ],
        "missing_required_tools": [
            tool.get("tool_id") for tool in tool_plan.get("missing_required_tools", [])
        ],
        "missing_optional_tools": [
            tool.get("tool_id") for tool in tool_plan.get("missing_optional_tools", [])
        ],
        "clinical_doc_ids": tool_plan.get("clinical_doc_ids", []),
        "output_schema": tool_plan.get("output_schema"),
    }


def build_policy_retrieval_query(condition_policy: dict[str, Any]) -> str:
    condition_name = condition_policy.get("condition_name") or "Unknown condition"
    severity_axes = ", ".join(condition_policy.get("primary_severity_axes", [])) or "severity"
    network_features = ", ".join(condition_policy.get("network_features", []))
    tool_features = ", ".join(condition_policy.get("required_tools", []))
    feature_clause = ", ".join(filter(None, [network_features, tool_features]))
    if feature_clause:
        return (
            f"{condition_name} severity or staging criteria using {severity_axes}; "
            f"available feature sources: {feature_clause}"
        )
    return f"{condition_name} severity or staging criteria using {severity_axes}"


def _tool_metadata(tool_id: str, tool_registry: dict[str, Any]) -> dict[str, Any]:
    return tool_registry.get(
        tool_id,
        {
            "implementation_status": "not_implemented",
            "runtime_tool": None,
            "limitations": ["Tool is referenced by policy but not defined in the tool registry."],
        },
    )


def _is_runtime_tool(tool: dict[str, Any]) -> bool:
    return tool.get("implementation_status") == "implemented" and bool(tool.get("runtime_tool"))


def _runtime_tool_entry(tool_id: str, tool: dict[str, Any], source: str) -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "runtime_tool": tool.get("runtime_tool"),
        "source": source,
        "feature_outputs": tool.get("feature_outputs", []),
        "limitations": tool.get("limitations", []),
    }


def _missing_tool_entry(tool_id: str, tool: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_id": tool_id,
        "implementation_status": tool.get("implementation_status", "not_implemented"),
        "limitations": tool.get("limitations", []),
    }
