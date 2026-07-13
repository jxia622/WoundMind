import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from app.config import FEEDBACK_LOG_PATH


def append_feedback(payload: dict[str, Any], log_path: Path = FEEDBACK_LOG_PATH) -> dict:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        **payload,
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(record) + "\n")

    return {
        "saved": True,
        "path": str(log_path),
    }
