# WoundMind UI

Two-page wound analysis interface connected to the local FastAPI model pipeline.
It runs condition classification, segmentation, full-frame depth preview, and
routed DFU/pressure-injury severity inference.

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
