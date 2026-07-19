from __future__ import annotations

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app.agent.config as agent_config_module
import app.agent.tools as agent_tools_module
from app.agent.config import AgentConfig
from app.agent.schemas import DocumentChunk
from app.main import app


class FakeSegmentationModel:
    def __init__(self) -> None:
        self.calls: list[Image.Image] = []

    def predict(self, image: Image.Image) -> dict:
        self.calls.append(image)
        mask = Image.new("L", image.size, 200)
        return {"mask_available": True, "mask": mask, "mask_source": "fake-unet", "warnings": []}


class FakeDepthModel:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

    def predict(self, image: Image.Image, mask: Image.Image | None = None, **kwargs) -> dict:
        self.calls.append((image, mask))
        depth = Image.new("L", image.size, 128)
        return {
            "depth_available": True,
            "depth": depth,
            "depth_full_preview": depth,
            "depth_source": "fake-depth-anything",
            "depth_scale": "relative",
            "roi": None,
            "warnings": [],
        }


class FakePipeline:
    def __init__(
        self,
        condition_top1_label: str,
        condition_confidence: float = 0.91,
        severity_available: bool = True,
        severity_stage: str | None = "Grade 3",
        severity_confidence: float = 0.88,
    ) -> None:
        self.segmentation_model = FakeSegmentationModel()
        self.depth_model = FakeDepthModel()
        self._condition_top1_label = condition_top1_label
        self._condition_confidence = condition_confidence
        self._severity_available = severity_available
        self._severity_stage = severity_stage
        self._severity_confidence = severity_confidence
        self.predict_severity_calls: list[tuple] = []

    def validate_image(self, image: Image.Image) -> dict:
        return {"warnings": [], "image_ok": True}

    def predict_condition(self, image: Image.Image) -> dict:
        return {
            "case_id": "fake-case",
            "top1_label": self._condition_top1_label,
            "confidence": self._condition_confidence,
            "confidence_level": "high",
            "uncertainty_level": "low",
            "top3": [{"label": self._condition_top1_label, "probability": self._condition_confidence}],
            "top2_probability_margin": 0.5,
            "normalized_entropy": 0.1,
        }

    def predict_severity(
        self,
        image: Image.Image,
        condition: str,
        condition_source: str,
        mask_status: str,
        depth_status: str,
        mask: Image.Image | None = None,
        depth: Image.Image | None = None,
    ) -> dict:
        self.predict_severity_calls.append((condition, mask_status, depth_status))
        if not self._severity_available:
            return {
                "severity_available": False,
                "selected_condition": condition,
                "message": "Severity model for this condition is still under development.",
            }
        return {
            "severity_available": True,
            "selected_condition": condition,
            "severity_prediction": self._severity_stage,
            "severity_confidence": self._severity_confidence,
            "model_used": "fake-severity-model",
            "input_channels_used": ["rgb", "mask", "depth"],
            "class_probabilities": {self._severity_stage: self._severity_confidence},
            "mock_output": False,
            "warning": None,
        }


class FakePubMedMCPRetriever:
    instances: list["FakePubMedMCPRetriever"] = []

    def __init__(self, config) -> None:
        self.config = config
        self.calls: list[tuple] = []
        FakePubMedMCPRetriever.instances.append(self)

    async def retrieve(self, query: str, condition: str | None, top_k: int = 4):
        self.calls.append((query, condition, top_k))
        return [
            DocumentChunk(
                text=(
                    "Wagner grade 3 diabetic foot ulcers show deep ulceration with "
                    "possible abscess or osteomyelitis."
                ),
                source="https://pubmed.ncbi.nlm.nih.gov/999999/",
                doc_id="PMID:999999",
                section="PubMed abstract",
                condition_tag="dfu",
                score=0.95,
            )
        ]


@pytest.fixture(autouse=True)
def _reset_fake_pubmed_instances():
    FakePubMedMCPRetriever.instances = []
    yield
    FakePubMedMCPRetriever.instances = []


def _configure_isolated_agent_env(monkeypatch, tmp_path) -> None:
    # Never read the developer's real .env: guarantees OPENAI_API_KEY and
    # LITERATURE_RETRIEVAL_BACKEND stay whatever the test sets, regardless of
    # what's on disk locally, and guarantees no live OpenAI/PubMed call.
    monkeypatch.setattr(agent_config_module, "load_project_env", lambda *args, **kwargs: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LITERATURE_RETRIEVAL_BACKEND", raising=False)
    monkeypatch.setenv("CASE_ARTIFACTS_PATH", str(tmp_path / "case_artifacts"))
    monkeypatch.setattr(agent_tools_module, "PubMedMCPRetriever", FakePubMedMCPRetriever)


def _make_test_image_bytes() -> io.BytesIO:
    image = Image.new("RGB", (256, 256), color=(180, 90, 90))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    buffer.seek(0)
    return buffer


def _post_analyze(client: TestClient, pipeline, **form_fields):
    client.app.state.pipeline = pipeline
    client.app.state.startup_error = None
    return client.post(
        "/agent/analyze",
        files={"image": ("wound.jpg", _make_test_image_bytes(), "image/jpeg")},
        data=form_fields,
    )


def test_agent_analyze_dfu_condition_override_runs_full_tool_plan(monkeypatch, tmp_path):
    _configure_isolated_agent_env(monkeypatch, tmp_path)
    fake_pipeline = FakePipeline(
        condition_top1_label="Healthy_Normal_Skin",
        severity_available=True,
        severity_stage="Grade 3",
    )
    client = TestClient(app)

    response = _post_analyze(
        client,
        fake_pipeline,
        selected_condition="Diabetic_Foot_Ulcers_(DFU)",
    )

    assert response.status_code == 200
    payload = response.json()
    output = payload["output"]
    assert output["condition"] == "Diabetic_Foot_Ulcers_(DFU)"
    assert output["severity_stage"] == "Grade 3"

    trace_tools = [step["tool"] for step in output["agent_trace"]]
    assert "override_condition" in trace_tools
    assert "segment_wound" in trace_tools
    assert "depth_map" in trace_tools
    assert len(fake_pipeline.segmentation_model.calls) == 1
    assert len(fake_pipeline.depth_model.calls) == 1

    assert len(FakePubMedMCPRetriever.instances) == 1
    retriever = FakePubMedMCPRetriever.instances[0]
    assert len(retriever.calls) == 2  # initial policy query + verification query
    assert payload["retrieved_chunks"][0]["doc_id"] == "PMID:999999"


def test_agent_analyze_non_dfu_pi_condition_stops_after_classification(monkeypatch, tmp_path):
    _configure_isolated_agent_env(monkeypatch, tmp_path)
    fake_pipeline = FakePipeline(
        condition_top1_label="Acne",
        severity_available=False,
        severity_stage=None,
    )
    client = TestClient(app)

    response = _post_analyze(client, fake_pipeline)

    assert response.status_code == 200
    output = response.json()["output"]
    assert output["condition"] == "Acne"
    assert output["severity_stage"] is None
    assert "Severity stage suppressed because condition is outside DFU/PI." in output["flags"]

    assert fake_pipeline.segmentation_model.calls == []
    assert fake_pipeline.depth_model.calls == []
    trace_tools = [step["tool"] for step in output["agent_trace"]]
    assert "segment_wound" not in trace_tools
    assert "depth_map" not in trace_tools
    assert trace_tools.count("skip_tool") == 2


def test_agent_analyze_without_openai_key_uses_deterministic_verifier_fallback(monkeypatch, tmp_path):
    _configure_isolated_agent_env(monkeypatch, tmp_path)
    assert AgentConfig.from_env().openai_enabled is False

    fake_pipeline = FakePipeline(
        condition_top1_label="Diabetic_Foot_Ulcers_(DFU)",
        severity_available=True,
        severity_stage="Grade 3",
    )
    client = TestClient(app)

    response = _post_analyze(client, fake_pipeline)

    assert response.status_code == 200
    output = response.json()["output"]
    assert output["verifier_result"] == "PASS"
    assert output["citation"] is not None
    assert "Deterministic local verifier used because OPENAI_API_KEY is not set." in output["flags"]
