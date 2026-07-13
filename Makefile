.PHONY: setup api streamlit static-demo start-demo status-demo stop-demo validate test compile

PYTHON ?= python3

setup:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install --upgrade pip
	.venv/bin/pip install -r requirements.txt -r requirements_agent.txt

api:
	.venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

streamlit:
	.venv/bin/streamlit run frontend/streamlit_app.py

static-demo:
	cd frontend/static-demo && ../../.venv/bin/python -m http.server 4173 --bind 127.0.0.1

start-demo:
	./scripts/dev/start_demo.sh

status-demo:
	./scripts/dev/status_demo.sh

stop-demo:
	./scripts/dev/stop_demo.sh

validate: compile
	$(PYTHON) -m json.tool app/condition_class_map.json >/dev/null
	$(PYTHON) -m json.tool app/temperature_calibration.json >/dev/null
	$(PYTHON) -m json.tool app/agent/condition_policy_registry.json >/dev/null
	$(PYTHON) -m json.tool clinical_docs/clinical_reference_download_manifest.json >/dev/null
	zsh -n scripts/dev/start_demo.sh scripts/dev/status_demo.sh scripts/dev/stop_demo.sh

test:
	$(PYTHON) -m pytest tests

compile:
	$(PYTHON) -m compileall app frontend scripts tests
