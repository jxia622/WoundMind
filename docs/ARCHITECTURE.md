# Architecture

WoundMind is organized as a diagnostic assistant around a deterministic tool pipeline. The current static demo exposes a one-way review workflow: image quality verification, condition classification with optional override, side-by-side segmentation/depth review, and a full diagnostic-agent evaluation with trace and evidence metadata.

## System Components

```mermaid
flowchart TB
    UI["Static demo / Streamlit"] --> API["FastAPI service"]
    API --> Pipeline["WoundAnalysisPipeline"]
    API --> Agent["WoundMindAgent"]

    Pipeline --> Condition["ConditionModel"]
    Pipeline --> Segment["SegmentationModel"]
    Pipeline --> Depth["DepthModel"]
    Pipeline --> Severity["SeverityModelManager"]

    Agent --> Tools["AgentTools"]
    Agent --> Policy["condition_policy_registry.json"]
    Agent --> Evaluation["Blinded LangGraph evaluation"]
    Agent --> Logger["CaseArtifactLogger"]

    Tools --> Pipeline
    Evaluation --> Verbalizer["Initial / targeted Verbalizer"]
    Evaluation --> PubAgent["Existing PubAgent"]
    Evaluation --> Adjudicator["Post-reveal adjudicator"]
```

## Runtime Flow

```mermaid
flowchart TB
    A["Upload image"] --> B["Validate quality"]
    B --> C["Predict condition top-k"]
    C --> D["User can override condition"]
    D --> E{"Supported severity route?"}
    E -- "No" --> F["Stop after classification"]
    E -- "DFU / PI" --> G["Generate segmentation mask"]
    G --> H["Render light-blue mask overlay on original image"]
    E -- "DFU / PI" --> I["Generate relative depth map"]
    I --> J["Render depth map beside depth legend"]
    H --> K["Run WoundMindAgent"]
    J --> K
    K --> L["Load condition policy and run required tools"]
    L --> M["Complete deterministic condition + severity output"]
    M --> N["Prepare blinded evidence state"]
    N --> O["Verbalizer + bounded Orchestrator loop"]
    O --> P["Commit independent assessment"]
    P --> Q["Reveal deterministic result"]
    Q --> R["Compare and return disposition + audit trace"]
```

1. The API loads `WoundAnalysisPipeline` at startup.
2. The uploaded image is validated and normalized to RGB.
3. The condition classifier returns top-k condition probabilities and confidence metadata.
4. The static demo exposes the condition as a dropdown so a reviewer can override the model label before downstream routing.
5. DFU and pressure-injury routes continue to segmentation, depth, severity routing, and agent evaluation; unsupported conditions stop after classification.
6. Segmentation is shown as a light-blue overlay on the original wound image; the relative depth map is shown beside the depth legend.
7. After deterministic inference completes, the evaluator receives only wound evidence. Structured Orchestrator actions can ask the Verbalizer for case-specific facts or the existing PubAgent for clinical evidence.
8. The evaluator commits and locks its independent assessment before the deterministic condition, stage, and confidence are revealed.
9. Adjudication returns `SUPPORTED`, `FLAGGED`, or `INSUFFICIENT_EVIDENCE`; the full graph state is checkpointed in process and its audit record is persisted with the case artifact.

## Agent Boundary

The current agent layer is deliberately conservative:

- It can run without an LLM key, in which case the independent result is conservatively `INSUFFICIENT_EVIDENCE`.
- Tool execution remains deterministic and auditable.
- Clinical-literature questions are delegated only to the existing PubAgent interface.
- The simulated Q&A helper only answers from provided context.
- Conditions outside the currently implemented DFU/pressure-injury severity routes are not forced through unsupported staging tools.

OpenAI-backed Verbalizer, Orchestrator, and adjudicator calls use schema-enforced outputs. Blinded contexts have no model-result fields, and the independent-assessment digest is checked after reveal. See [EVALUATION_ARCHITECTURE.md](EVALUATION_ARCHITECTURE.md).

## Model Artifacts

The repository tracks model-loading code and checkpoint placement docs, not the binary checkpoints. See [CHECKPOINTS.md](CHECKPOINTS.md) for the expected directory layout.
