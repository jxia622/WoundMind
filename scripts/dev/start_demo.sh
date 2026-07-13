#!/bin/zsh

set -eu

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h:h}"
RUNTIME_DIR="$PROJECT_DIR/.demo-runtime"
LOG_DIR="$RUNTIME_DIR/logs"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_DIR/.venv/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3)"
fi

mkdir -p "$LOG_DIR"

if [[ -f "$PROJECT_DIR/.env" ]]; then
  set -a
  source "$PROJECT_DIR/.env"
  set +a
fi

(
  cd "$PROJECT_DIR"
  "$PYTHON_BIN" -m uvicorn app.main:app --host 127.0.0.1 --port 8000 \
    > "$LOG_DIR/backend.log" 2> "$LOG_DIR/backend-error.log" &
  echo $! > "$RUNTIME_DIR/backend.pid"
)

(
  cd "$PROJECT_DIR/frontend/static-demo"
  "$PYTHON_BIN" -m http.server 4173 --bind 127.0.0.1 \
    > "$LOG_DIR/frontend.log" 2> "$LOG_DIR/frontend-error.log" &
  echo $! > "$RUNTIME_DIR/frontend.pid"
)

echo "Waiting for the model API..."
for attempt in {1..90}; do
  if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
    echo "WoundMind is ready: http://localhost:4173"
    exit 0
  fi
  sleep 1
done

echo "Backend did not become ready."
echo "Check $LOG_DIR/backend.log and $LOG_DIR/backend-error.log"
exit 1
