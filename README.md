# WoundMind

WoundMind is a research prototype for a wound diagnostic assistant agent. It combines a multi-stage image analysis pipeline with human-in-the-loop review and clinical reference retrieval so wound assessment can be more traceable than a single black-box prediction.

> Research prototype. Not for clinical diagnosis.

## What is implemented

- FastAPI backend for image validation, condition prediction, mask generation, depth estimation, severity routing, agent analysis, and feedback logging.
- Streamlit workflow for interactive local use.
- Static browser UI prototype in `frontend/static-demo/`.
- ConvNeXt-Tiny condition classifier for 18 skin and wound conditions.
- U-Net++ wound segmentation wrapper for mask generation.
- Depth Anything V2 wrapper for relative depth estimation.
- DFU and pressure injury severity routing across RGB, RGB+mask, RGB+depth, and RGB+depth+mask model variants.
- Agent modules for tool orchestration, policy lookup, clinician-question simulation, verifier state, case logging, and clinical document retrieval.
- Local clinical reference index in `chroma_db/wound_clinical_docs.json`.

## Repository layout

```text
app/                    FastAPI app, pipeline, models, routing, agent modules
frontend/static-demo/   Static browser prototype for the local demo
scripts/                Clinical document ingestion utilities
chroma_db/              Local JSON clinical reference index
clinical_docs/          Source manifests and text references; PDFs stay local
docs/                   Project notes and checkpoint placement instructions
tests                   Root-level smoke and preprocessing tests
```

## Model checkpoints

Large model checkpoints are not committed to git. Put them under `checkpoints/` using the paths listed in [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements_agent.txt
```

Depth Anything V2 is loaded from Hugging Face with the model id `depth-anything/Depth-Anything-V2-Base-hf`. It is not stored in this repo.

## Run

Start the API and static prototype:

```bash
./start_demo.sh
```

Then open:

- API: http://localhost:8000
- Swagger: http://localhost:8000/docs
- Static demo: http://localhost:4173

For the Streamlit workflow:

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

## Validation

```bash
python -m compileall app
python -m pytest test_preprocessing.py
```

The full inference tests require local checkpoints.

## Not Yet Built

- Full clinical validation against an independently reviewed wound dataset.
- Production-grade deployment, authentication, access control, audit logging, and PHI-safe storage.
- Fully automated clinician-facing follow-up question flow beyond the current simulated/context-driven Q&A helper.
- A polished production UI that unifies the Streamlit workflow and static prototype.
- Remote model artifact hosting and reproducible checkpoint download scripts.
- Prospective safety evaluation, calibration review, and clinician sign-off workflow.
