from __future__ import annotations

from uuid import uuid4

from PIL import Image

from app.agent.case_logger import CaseArtifactLogger
from app.agent.config import AgentConfig
from app.agent.evaluation import EvaluationAgentLoop
from app.agent.memory import ClinicalMemory
from app.agent.schemas import AgentAnalyzeResponse, ClinicalOutput, QAExchange
from app.agent.state import AgentState
from app.agent.tool_policy import (
    build_policy_retrieval_query,
    build_tool_plan,
    get_condition_policy,
    load_condition_policy_registry,
    runtime_tool_names,
    summarize_tool_plan,
)
from app.agent.tools import AgentTools
from app.config import DISCLAIMER


class WoundMindAgent:
    """
    First-pass agent implementation around the existing WoundAnalysisPipeline.

    The current code keeps the model sequence deterministic so it can run without
    an LLM key. GPT-4o brain/verifier paths can be expanded behind this same
    interface without changing the API route.
    """

    def __init__(
        self,
        pipeline,
        config: AgentConfig | None = None,
        memory: ClinicalMemory | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.config = config or AgentConfig.from_env()
        self.memory = memory or (
            ClinicalMemory(self.config) if self.config.literature_backend == "local" else None
        )
        self.tools = AgentTools(pipeline=pipeline, config=self.config, memory=self.memory)
        self.evaluation_loop = EvaluationAgentLoop(self.config)
        self.case_logger = CaseArtifactLogger(self.config)
        self.policy_registry = load_condition_policy_registry()

    async def analyze(
        self,
        image: Image.Image,
        patient_context: dict | None = None,
        case_id: str | None = None,
        condition_override: str | None = None,
    ) -> AgentAnalyzeResponse:
        state = AgentState(
            case_id=case_id or str(uuid4()),
            patient_context=patient_context or {},
            input_image=image.convert("RGB"),
        )

        state.validation_result = self.pipeline.validate_image(image)
        state.add_trace("validate_image", {}, state.validation_result)

        state.classification_result = await self.tools.classify_condition(image)
        condition = state.classification_result.get("top1_label")
        condition_confidence = state.classification_result.get("confidence")
        state.add_trace(
            "classify_condition",
            {"image": "uploaded"},
            {
                "top1_label": condition,
                "confidence": condition_confidence,
                "top3": state.classification_result.get("top3"),
            },
        )

        if condition_override and condition_override != condition:
            state.add_trace(
                "override_condition",
                {"model_top1": condition},
                {"selected_condition": condition_override},
            )
            state.classification_result["model_top1_label"] = condition
            state.classification_result["top1_label"] = condition_override
            condition = condition_override

        state.condition_policy = get_condition_policy(condition, self.policy_registry)
        state.tool_plan = build_tool_plan(state.condition_policy, self.policy_registry)
        state.add_trace(
            "load_condition_policy",
            {"condition": condition},
            summarize_tool_plan(state.tool_plan),
        )

        allowed_doc_ids = state.condition_policy.get("clinical_doc_ids", [])
        initial_query = build_policy_retrieval_query(state.condition_policy)
        state.doc_chunks = await self.tools.retrieve_docs(
            initial_query,
            condition,
            allowed_doc_ids=allowed_doc_ids,
        )
        state.add_trace(
            "retrieve_docs",
            {
                "query": initial_query,
                "condition": condition,
                "allowed_doc_ids": allowed_doc_ids,
            },
            {"chunks": [chunk.model_dump(exclude={"text"}) for chunk in state.doc_chunks]},
        )

        planned_runtime_tools = runtime_tool_names(state.tool_plan)
        if "segment_wound" in planned_runtime_tools:
            segment_result = await self.tools.segment_wound(image)
            state.mask_result = segment_result["summary"]
            state.mask_image = segment_result["mask"]
            state.selected_mask_image = state.mask_image
            state.selected_mask = {
                "selected_mask_idx": 0 if state.mask_image is not None else None,
                "rationale": "Single U-Net++ mask available; selected for downstream depth and severity tools.",
            }
            state.add_trace("segment_wound", {"model": "U-Net++"}, state.mask_result)
            state.add_trace("select_mask", {"candidate_count": 1}, state.selected_mask)
        else:
            state.mask_result = self._skipped_tool_summary(
                "open_wound_segmentation",
                state.tool_plan,
            )
            state.selected_mask = {
                "selected_mask_idx": None,
                "rationale": "No policy-required implemented segmentation tool for this condition.",
            }
            state.add_trace(
                "skip_tool",
                {"runtime_tool": "segment_wound"},
                state.mask_result,
            )

        depth_payload = {"depth": None}
        if "depth_map" in planned_runtime_tools:
            depth_result = await self.tools.depth_map(image, state.selected_mask_image)
            depth_payload = depth_result
            state.depth_result = depth_result["summary"]
            state.depth_preview_image = depth_result["preview"]
            state.add_trace("depth_map", {"mask": "selected_mask"}, state.depth_result)
        else:
            state.depth_result = self._skipped_tool_summary("relative_depth_map", state.tool_plan)
            state.add_trace(
                "skip_tool",
                {"runtime_tool": "depth_map"},
                state.depth_result,
            )

        state.severity_result = await self.tools.predict_severity(
            image=image,
            condition=condition,
            mask=state.selected_mask_image,
            depth=depth_payload["depth"],
        )
        state.add_trace(
            "predict_severity",
            {
                "condition": condition,
                "mask_status": "accepted" if state.selected_mask_image else "none",
                "depth_status": "accepted" if depth_payload["depth"] is not None else "rejected",
            },
            self._severity_summary(state.severity_result),
        )

        await self._maybe_ask_questions(state)
        stage = self._stage_for_output(state)
        state.visual_evidence = await self.tools.extract_visual_evidence(
            image=image,
            condition=condition,
            stage=stage,
            mask_summary=state.mask_result,
            depth_summary=state.depth_result,
        )
        state.add_trace(
            "extract_visual_evidence",
            {"condition": condition, "stage": stage, "image": "uploaded"},
            self._visual_evidence_summary(state.visual_evidence),
        )

        draft = {
            "condition": condition,
            "condition_confidence": condition_confidence,
            "severity_stage": stage,
            "severity_confidence": state.severity_result.get("severity_confidence")
            if state.severity_result
            else None,
            "rationale": self._draft_rationale(state),
            "visual_evidence": state.visual_evidence,
        }
        state.draft_assessment = draft

        verification_query = f"{condition} {stage or ''} staging criteria"
        state.doc_chunks = await self.tools.retrieve_docs(
            verification_query,
            condition,
            allowed_doc_ids=allowed_doc_ids,
        )
        state.add_trace(
            "retrieve_docs",
            {
                "query": verification_query,
                "condition": condition,
                "allowed_doc_ids": allowed_doc_ids,
            },
            {"chunks": [chunk.model_dump(exclude={"text"}) for chunk in state.doc_chunks]},
        )

        evaluation = await self.evaluation_loop.run(
            base_draft=draft,
            visual_evidence=state.visual_evidence or {},
            doc_chunks=state.doc_chunks,
            condition=condition,
            retrieve_docs=self.tools.retrieve_docs,
        )
        state.draft_assessment = evaluation.draft
        state.doc_chunks = evaluation.doc_chunks
        state.verifier_result = evaluation.verifier_result
        state.evaluation_summary = evaluation.summary
        for step in evaluation.trace_steps:
            state.add_trace(
                step["tool"],
                step.get("inputs_summary", {}),
                step.get("outputs_summary", {}),
            )
        state.add_trace(
            "verify_assessment",
            {"draft": state.draft_assessment, "mode": "agent_evaluator_loop"},
            state.verifier_result.model_dump(),
        )

        output = self._final_output(state)
        artifact_path = self.case_logger.write_case(state, output)
        output.artifact_path = str(artifact_path)
        (artifact_path / "clinical_output.json").write_text(output.model_dump_json(indent=2))

        return AgentAnalyzeResponse(
            case_id=state.case_id,
            output=output,
            retrieved_chunks=state.doc_chunks,
        )

    async def _maybe_ask_questions(self, state: AgentState) -> None:
        condition = state.classification_result.get("top1_label") if state.classification_result else ""
        condition_confidence = (
            state.classification_result.get("confidence") if state.classification_result else None
        )
        severity_confidence = (
            state.severity_result.get("severity_confidence") if state.severity_result else None
        )
        questions: list[str] = []
        if condition_confidence is not None and condition_confidence < self.config.confidence_threshold:
            questions.append("How long has this wound been present?")
        if self._is_dfu_or_pi(condition) and "is_diabetic" not in state.patient_context:
            questions.append("Is this patient diabetic?")
        if severity_confidence is not None and severity_confidence < self.config.severity_threshold:
            questions.append("Is there exposed bone or tendon visible?")
        if "prior_treatment" not in state.patient_context:
            questions.append("Has debridement or other wound treatment been performed previously?")

        for question in questions[: self.config.max_qa]:
            answer = await self.tools.ask_clinician(question, state.patient_context)
            exchange = QAExchange(question=question, answer=answer)
            state.qa_history.append(exchange)
            state.add_trace("ask_clinician", {"question": question}, {"answer": answer})

    def _final_output(self, state: AgentState) -> ClinicalOutput:
        verifier = state.verifier_result
        flags: list[str] = []
        if verifier and verifier.flag:
            flags.append(verifier.flag)
        if state.validation_result:
            flags.extend(state.validation_result.get("warnings", []))
        if state.severity_result and not state.severity_result.get("severity_available", True):
            flags.append(state.severity_result.get("message", "Severity unavailable."))
        if state.tool_plan:
            for tool in state.tool_plan.get("missing_required_tools", []):
                flags.append(
                    f"Required tool not implemented for {state.tool_plan.get('condition_id')}: "
                    f"{tool.get('tool_id')}."
                )
            policy_doc_ids = state.tool_plan.get("clinical_doc_ids", [])
            if policy_doc_ids and not state.doc_chunks:
                if self.config.literature_backend == "local":
                    flags.append(
                        "No indexed clinical chunks matched policy document IDs: "
                        f"{', '.join(policy_doc_ids)}."
                    )
                else:
                    flags.append(
                        "No PubMed MCP literature results were retrieved for the policy query."
                    )

        stage = self._stage_for_output(state)
        if verifier and verifier.result == "UNCERTAIN":
            stage = None
            flags.append("Verifier returned UNCERTAIN; staging recommendation suppressed.")
        if not self._is_dfu_or_pi(
            state.classification_result.get("top1_label") if state.classification_result else ""
        ):
            stage = None
            flags.append("Severity stage suppressed because condition is outside DFU/PI.")

        return ClinicalOutput(
            case_id=state.case_id,
            condition=state.classification_result.get("top1_label") if state.classification_result else None,
            condition_confidence=state.classification_result.get("confidence")
            if state.classification_result
            else None,
            severity_stage=stage,
            severity_confidence=state.severity_result.get("severity_confidence")
            if state.severity_result and stage
            else None,
            recommendation=self._recommendation(state, stage),
            verifier_result=verifier.result if verifier else "UNCERTAIN",
            citation=verifier.citation if verifier and verifier.result == "PASS" else None,
            flags=flags,
            visual_evidence=state.visual_evidence or {},
            evaluation_summary=state.evaluation_summary or {},
            qa_exchanges=state.qa_history,
            agent_trace=state.trace,
            model_variant_used=state.severity_result.get("model_used") if state.severity_result else None,
            disclaimer=DISCLAIMER,
        )

    @staticmethod
    def _skipped_tool_summary(tool_id: str, tool_plan: dict | None) -> dict:
        missing_required = {
            tool.get("tool_id") for tool in (tool_plan or {}).get("missing_required_tools", [])
        }
        if tool_id in missing_required:
            reason = "Tool is required by the condition policy but is not implemented."
        else:
            reason = "Tool is not required by the condition policy for this condition."

        return {
            "tool_id": tool_id,
            "tool_skipped": True,
            "condition_id": (tool_plan or {}).get("condition_id"),
            "reason": reason,
        }

    @staticmethod
    def _severity_summary(severity_result: dict | None) -> dict:
        if not severity_result:
            return {}
        keys = [
            "severity_available",
            "selected_condition",
            "severity_prediction",
            "severity_confidence",
            "model_used",
            "input_channels_used",
            "message",
            "warning",
        ]
        return {key: severity_result.get(key) for key in keys if key in severity_result}

    @staticmethod
    def _visual_evidence_summary(visual_evidence: dict | None) -> dict:
        if not visual_evidence:
            return {}
        return {
            "vlm_used": visual_evidence.get("vlm_used", False),
            "source": visual_evidence.get("source"),
            "model": visual_evidence.get("model"),
            "wound_visible": visual_evidence.get("wound_visible"),
            "image_quality": visual_evidence.get("image_quality"),
            "findings": visual_evidence.get("findings", [])[:5],
            "staging_relevant_observations": visual_evidence.get(
                "staging_relevant_observations", []
            )[:5],
            "limitations": visual_evidence.get("limitations", [])[:5],
        }

    @staticmethod
    def _stage_for_output(state: AgentState) -> str | None:
        if not state.severity_result or not state.severity_result.get("severity_available"):
            return None
        return state.severity_result.get("severity_prediction")

    @staticmethod
    def _is_dfu_or_pi(condition: str | None) -> bool:
        normalized = (condition or "").lower()
        return "diabetic_foot" in normalized or "dfu" in normalized or "pressure" in normalized

    @staticmethod
    def _draft_rationale(state: AgentState) -> str:
        condition = state.classification_result.get("top1_label") if state.classification_result else "unknown"
        severity = state.severity_result or {}
        if severity.get("severity_available"):
            return (
                f"Existing model pipeline predicted {condition}; routed severity model "
                f"{severity.get('model_used')} returned {severity.get('severity_prediction')}."
            )
        return f"Existing model pipeline predicted {condition}; severity is not available for this condition."

    @staticmethod
    def _recommendation(state: AgentState, stage: str | None) -> str:
        condition = state.classification_result.get("top1_label") if state.classification_result else "unknown"
        if stage is None:
            return (
                "Escalate to a qualified clinician for review. The agent did not produce a "
                "verified staging recommendation."
            )
        return (
            f"Review the model-suggested {condition} severity ({stage}) with a qualified "
            "clinician and use local wound-care protocol for next steps."
        )
