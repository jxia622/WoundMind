from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.agent.config import AgentConfig  # noqa: E402
from app.agent.memory import ClinicalMemory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest clinical reference PDFs/text into the WoundMind agent retrieval index."
    )
    parser.add_argument(
        "--docs-dir",
        type=Path,
        default=None,
        help="Defaults to CLINICAL_DOCS_PATH or ./clinical_docs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = AgentConfig.from_env()
    memory = ClinicalMemory(config)
    summary = memory.ingest_documents(args.docs_dir or config.clinical_docs_path)
    print("WoundMind clinical document ingestion complete.")
    for key, value in summary.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
