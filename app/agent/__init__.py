from app.agent.config import AgentConfig
from app.agent.schemas import ClinicalOutput, VerifierResult

__all__ = [
    "AgentConfig",
    "ClinicalOutput",
    "VerifierResult",
    "WoundMindAgent",
]


def __getattr__(name: str):
    if name == "WoundMindAgent":
        from app.agent.brain import WoundMindAgent

        return WoundMindAgent
    raise AttributeError(f"module 'app.agent' has no attribute {name!r}")
