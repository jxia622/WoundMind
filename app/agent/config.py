from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from app.config import PROJECT_ROOT


def load_project_env(path: Path | None = None) -> None:
    env_path = path or PROJECT_ROOT / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and key not in os.environ:
            os.environ[key] = value


@dataclass(frozen=True)
class AgentConfig:
    model_provider: str
    agent_model: str
    verifier_model: str
    embedding_model: str
    literature_backend: str
    pubmed_mcp_project_path: Path
    pubmed_mcp_python: str
    pubmed_mcp_timeout_seconds: int
    pubmed_mcp_max_results: int
    chroma_db_path: Path
    clinical_docs_path: Path
    case_artifacts_path: Path
    max_iterations: int
    max_qa: int
    max_verifier_retries: int
    confidence_threshold: float
    severity_threshold: float
    timeout_seconds: int
    use_simulated_qa: bool
    vision_model: str = "gpt-4o"
    evaluation_model: str = "gpt-4o"

    @classmethod
    def from_env(cls) -> "AgentConfig":
        load_project_env()
        return cls(
            model_provider=os.getenv("AGENT_MODEL_PROVIDER", "openai"),
            agent_model=os.getenv("AGENT_MODEL", "gpt-4o"),
            verifier_model=os.getenv("VERIFIER_MODEL", "gpt-4o"),
            vision_model=os.getenv("VISION_MODEL", os.getenv("AGENT_MODEL", "gpt-4o")),
            evaluation_model=os.getenv(
                "EVALUATION_MODEL",
                os.getenv("VERIFIER_MODEL", "gpt-4o"),
            ),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
            literature_backend=os.getenv("LITERATURE_RETRIEVAL_BACKEND", "pubmed_mcp"),
            pubmed_mcp_project_path=Path(
                os.getenv(
                    "PUBMED_MCP_PROJECT_PATH",
                    str(PROJECT_ROOT.parent / "pubmed-clinical-mcp"),
                )
            ),
            pubmed_mcp_python=os.getenv("PUBMED_MCP_PYTHON") or sys.executable,
            pubmed_mcp_timeout_seconds=int(os.getenv("PUBMED_MCP_TIMEOUT_SECONDS", "45")),
            pubmed_mcp_max_results=int(os.getenv("PUBMED_MCP_MAX_RESULTS", "8")),
            chroma_db_path=Path(os.getenv("CHROMA_DB_PATH", str(PROJECT_ROOT / "chroma_db"))),
            clinical_docs_path=Path(
                os.getenv("CLINICAL_DOCS_PATH", str(PROJECT_ROOT / "clinical_docs"))
            ),
            case_artifacts_path=Path(
                os.getenv("CASE_ARTIFACTS_PATH", str(PROJECT_ROOT / "case_artifacts"))
            ),
            max_iterations=int(os.getenv("AGENT_MAX_ITERATIONS", "8")),
            max_qa=int(os.getenv("AGENT_MAX_QA", "3")),
            max_verifier_retries=int(os.getenv("AGENT_MAX_VERIFIER_RETRIES", "2")),
            confidence_threshold=float(os.getenv("AGENT_CONFIDENCE_THRESHOLD", "0.75")),
            severity_threshold=float(os.getenv("AGENT_SEVERITY_THRESHOLD", "0.70")),
            timeout_seconds=int(os.getenv("AGENT_TIMEOUT_SECONDS", "120")),
            use_simulated_qa=os.getenv("AGENT_SIMULATED_QA", "true").lower()
            in {"1", "true", "yes", "on"},
        )

    @property
    def openai_enabled(self) -> bool:
        return self.model_provider == "openai" and bool(os.getenv("OPENAI_API_KEY"))
