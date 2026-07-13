#!/bin/zsh

PROJECT_DIR="${0:A:h}"
RUNTIME_DIR="$PROJECT_DIR/.demo-runtime"

for service in backend frontend; do
  pid_file="$RUNTIME_DIR/$service.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
    echo "$service: running (PID $(cat "$pid_file"))"
  else
    echo "$service: stopped"
  fi
done

if curl -fsS http://127.0.0.1:8000/health >/dev/null 2>&1; then
  echo "API health: ok"
else
  echo "API health: unavailable"
fi

if curl -fsS http://127.0.0.1:4173/ >/dev/null 2>&1; then
  echo "Frontend: available at http://localhost:4173"
else
  echo "Frontend: unavailable"
fi

echo "Logs: $RUNTIME_DIR/logs"
