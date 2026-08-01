# WoundMind Static Demo

One-way wound analysis interface connected to the local FastAPI model pipeline.
It verifies image quality, classifies the condition, allows condition override,
shows a light-blue segmentation overlay beside the depth map, and then runs the
two-pass blinded LangGraph evaluation for supported DFU/pressure-injury routes.
The interface keeps the deterministic model result separate from the independently
committed evaluator assessment and displays the final evaluation disposition.

## Run locally

Start FastAPI from the deployment project root:

```bash
source "/Users/jackxia/Desktop/Python/venv-hw/bin/activate"
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then start the frontend from this directory:

```bash
python3 -m http.server 4173
```

Open `http://localhost:4173`.

## URL options

Use `?api=http://127.0.0.1:8000` to override the backend URL.
