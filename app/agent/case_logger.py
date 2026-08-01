from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image

from app.agent.config import AgentConfig
from app.agent.schemas import ClinicalOutput
from app.agent.state import AgentState


class CaseArtifactLogger:
    def __init__(self, config: AgentConfig | None = None) -> None:
        self.config = config or AgentConfig.from_env()

    def write_case(self, state: AgentState, output: ClinicalOutput) -> Path:
        case_dir = self.config.case_artifacts_path / state.case_id
        case_dir.mkdir(parents=True, exist_ok=True)

        self._save_image(state.input_image, case_dir / "input.png")
        self._save_image(state.mask_image, case_dir / "generated_mask.png")
        self._save_image(state.selected_mask_image, case_dir / "selected_mask.png")
        self._save_image(state.depth_preview_image, case_dir / "depth_preview.png")

        self._write_json(
            case_dir / "predictions.json",
            {
                "validation": state.validation_result,
                "classification": state.classification_result,
                "condition_policy": state.condition_policy,
                "tool_plan": state.tool_plan,
                "severity": state.severity_result,
                "visual_evidence": state.visual_evidence,
                "evaluation_summary": state.evaluation_summary,
                "selected_mask": state.selected_mask,
            },
        )
        self._write_json(
            case_dir / "agent_trace.json",
            [step.model_dump() for step in state.trace],
        )
        self._write_json(
            case_dir / "verifier_output.json",
            state.verifier_result.model_dump() if state.verifier_result else None,
        )
        self._write_json(
            case_dir / "evaluation_output.json",
            state.evaluation_summary,
        )
        self._write_json(case_dir / "clinical_output.json", output.model_dump())
        return case_dir

    @staticmethod
    def _save_image(image: Image.Image | None, path: Path) -> None:
        if image is None:
            return
        image.save(path)

    @staticmethod
    def _write_json(path: Path, payload: Any) -> None:
        path.write_text(json.dumps(payload, indent=2, default=str))
