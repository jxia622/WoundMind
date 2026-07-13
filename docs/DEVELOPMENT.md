# Development Guide

## Setup

```bash
make setup
cp .env.example .env
```

Add local checkpoints before running full inference. The lightweight validation commands do not require checkpoint binaries.

## Common Commands

```bash
make api          # FastAPI only
make streamlit    # Streamlit workflow
make start-demo   # API + static demo
make status-demo
make stop-demo
make validate     # compile, JSON, shell-script checks
make test         # pytest suite
```

## Repo Conventions

- Keep runtime code in `app/`.
- Keep UI surfaces in `frontend/`.
- Keep local maintenance commands in `scripts/dev/`.
- Keep durable technical notes in `docs/`.
- Do not commit checkpoint binaries, downloaded PDFs, feedback logs, case artifacts, or local caches.

## Adding Clinical References

Place approved PDFs or text files in `clinical_docs/`, then run:

```bash
python scripts/ingest_docs.py
```

The generated or refreshed retrieval index lives in `chroma_db/`.
