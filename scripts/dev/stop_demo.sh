#!/bin/zsh

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h:h}"
RUNTIME_DIR="$PROJECT_DIR/.demo-runtime"

for service in backend frontend; do
  pid_file="$RUNTIME_DIR/$service.pid"
  if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" >/dev/null 2>&1; then
    kill "$(cat "$pid_file")"
    rm -f "$pid_file"
    echo "Stopped $service."
  else
    echo "$service was not running."
  fi
done
