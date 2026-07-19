from __future__ import annotations

import json
from types import SimpleNamespace

from PIL import Image

import app.agent.config as agent_config_module
from app.agent.config import AgentConfig
from app.agent.vision import VisualEvidenceExtractor


class FakeResponses:
    def __init__(self) -> None:
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            output_text=json.dumps(
                {
                    "vlm_used": True,
                    "source": "openai_vision",
                    "model": "fake-vision-model",
                    "wound_visible": True,
                    "image_quality": "usable",
                    "findings": ["visible ulcer bed"],
                    "staging_relevant_observations": ["deep-appearing tissue loss"],
                    "contraindicating_observations": [],
                    "limitations": ["single image only"],
                }
            )
        )


class FakeOpenAIClient:
    responses = FakeResponses()


def test_visual_evidence_extractor_calls_openai_with_input_image(monkeypatch):
    monkeypatch.setattr(agent_config_module, "load_project_env", lambda *args, **kwargs: None)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    fake_client = FakeOpenAIClient()
    monkeypatch.setitem(
        __import__("sys").modules,
        "openai",
        SimpleNamespace(OpenAI=lambda: fake_client),
    )

    config = AgentConfig.from_env()
    extractor = VisualEvidenceExtractor(config)
    image = Image.new("RGB", (64, 64), color=(160, 80, 80))

    evidence = extractor.extract(
        image,
        condition="Diabetic_Foot_Ulcers_(DFU)",
        stage="Grade 3",
        mask_summary={"mask_available": True},
        depth_summary={"depth_available": True},
    )

    assert evidence["vlm_used"] is True
    assert evidence["findings"] == ["visible ulcer bed"]
    call = fake_client.responses.calls[0]
    assert call["model"] == config.vision_model
    content = call["input"][0]["content"]
    assert any(item["type"] == "input_image" for item in content)
    assert any(item["type"] == "input_text" for item in content)
