# WoundMind

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)](#local-setup)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi&logoColor=white)](#run-locally)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](#run-locally)
[![Status](https://img.shields.io/badge/status-research%20prototype-orange)](#safety-note)

WoundMind is a diagnostic assistant agent for wound assessment research. The current demo runs a one-way image workflow: verify image quality, classify the wound condition with an override dropdown, show a light-blue segmentation overlay beside a relative depth map, then run a traceable full-agent evaluation for DFU and pressure-injury staging.

## Demo

![WoundMind DFU Grade 3 demo](docs/assets/woundmind_dfu_grade3_demo.gif)

[Watch the WoundMind DFU Grade 3 demo](docs/assets/woundmind_dfu_grade3_demo.webm)

The demo uses a diabetic foot ulcer image and shows the local UI returning a DFU condition prediction, Grade 3 severity result, confidence/probability summaries, a blue segmentation overlay on the original wound image, a side-by-side depth map, and a full diagnostic-agent trace with retrieved evidence and verifier output.

## Safety Note

WoundMind is a research prototype. It is not a medical device and is not for clinical diagnosis.

## What It Does

| Capability | Current implementation |
| --- | --- |
| Condition triage | ConvNeXt-Tiny classifier over 18 wound/skin condition classes |
| Wound localization | U-Net++ segmentation wrapper with light-blue mask overlay on the original wound image and mask-area summary |
| Depth signal | Depth Anything V2 relative-depth wrapper shown side-by-side with the segmentation overlay |
| Severity routing | DFU and pressure-injury severity model selection across RGB, mask, depth, and combined inputs |
| Agent orchestration | Full diagnostic-agent run with tool planner, policy registry, retrieval, verifier, trace steps, case logger, and simulated clinician Q&A |
| Clinical grounding | Local clinical reference index plus condition-gated retrieval policy |
| Human review | Static browser prototype and Streamlit workflow for condition override, visual review, and final agent output inspection |

## Agent Workflow

```mermaid
flowchart TB
    subgraph UI["Static demo: one-way review workflow"]
        A["Upload wound image"] --> B["Verify image quality"]
        B --> C["Classify condition"]
        C --> D["Condition dropdown override"]
        D --> E{"DFU or pressure injury?"}
        E -- "No" --> F["Stop after classification<br/>Other condition tools not built yet"]
        E -- "Yes" --> G["Generate U-Net++ mask"]
        G --> H["Show blue mask overlay<br/>on original image"]
        E -- "Yes" --> I["Generate Depth Anything V2 map"]
        I --> J["Show depth map beside legend"]
        H --> K["Run full agent evaluation"]
        J --> K
    end

    subgraph Agent["WoundMindAgent evaluation"]
        K --> L["Load condition policy<br/>and tool plan"]
        L --> M["Retrieve clinical references"]
        M --> N["Run segmentation and depth tools"]
        N --> O["Route DFU/PI severity model"]
        O --> P["Simulated clinician Q&A"]
        P --> Q["Verifier checks draft staging<br/>against retrieved evidence"]
        Q --> R["Final stage, probabilities,<br/>brief report, trace, evidence count"]
        R --> S["Log case artifacts"]
    end
```

The core agent path is deterministic and auditable. When `OPENAI_API_KEY` is set, the verifier can use OpenAI through the same interface; otherwise WoundMind falls back to deterministic local verification.

## Repository Layout

```text
app/
  main.py                  FastAPI service entrypoint
  pipeline.py              Multi-stage wound analysis pipeline
  models/                  Condition, segmentation, depth, severity wrappers
  agent/                   Tool planner, memory, verifier, policy, case logging
  routing/                 DFU/PI severity model routing rules
  preprocessing/           Image loading, transforms, validation

frontend/
  streamlit_app.py         Human-in-the-loop workflow UI
  static-demo/             Browser prototype for API-driven demo flow

scripts/
  ingest_docs.py           Clinical reference ingestion
  dev/                     Local demo and maintenance scripts

chroma_db/                 JSON fallback clinical reference index
clinical_docs/             Text references and source manifest; PDFs stay local
docs/                      Architecture, checkpoint, and development notes
tests/                     Smoke and preprocessing tests
```

## Local Setup

```bash
git clone https://github.com/jxia622/WoundMind.git
cd WoundMind
make setup
cp .env.example .env
```

Large model weights are not committed to git. Place them under `checkpoints/` using the paths in [docs/CHECKPOINTS.md](docs/CHECKPOINTS.md).

## Run Locally

Run the API and static browser prototype:

```bash
make start-demo
```

Open:

- API health: http://localhost:8000/health
- Swagger: http://localhost:8000/docs
- Static demo: http://localhost:4173

Run the Streamlit workflow:

```bash
make streamlit
```

Useful development commands:

```bash
make api
make status-demo
make stop-demo
make validate
```

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `POST /validate-image` | Basic image quality and format checks |
| `POST /predict-condition` | Top-k condition prediction |
| `POST /generate-mask` | U-Net++ wound mask generation |
| `POST /generate-depth` | Depth Anything V2 depth preview generation |
| `POST /predict-severity` | DFU/PI severity routing and inference |
| `POST /analyze` | Default end-to-end pipeline |
| `POST /agent/analyze` | Agent-mediated analysis with policy and retrieval context |
| `POST /feedback` | JSONL feedback logging |

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Checkpoint placement](docs/CHECKPOINTS.md)
- [Development guide](docs/DEVELOPMENT.md)

## Design References

The README structure follows common patterns used by mature agent repos: a compact product definition, capability table, quickstart, architecture diagram, API surface, and roadmap. The goal is to make the repo understandable before a reader opens the code.

## Not Yet Built

- Independent clinical validation with reviewed wound datasets and clinician-labeled outcomes.
- Production-grade authentication, authorization, PHI-safe storage, audit logging, and monitoring.
- Remote checkpoint hosting plus reproducible model-artifact download scripts.
- Fully automated clinician follow-up flow beyond the current simulated/context-driven Q&A helper.
- Unified production UI that replaces the split Streamlit and static demo surfaces.
- Prospective safety evaluation, calibration review, and clinician sign-off workflow.
