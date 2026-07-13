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
    Agent --> Memory["ClinicalMemory"]
    Agent --> Verifier["AssessmentVerifier"]
    Agent --> Logger["CaseArtifactLogger"]

    Tools --> Pipeline
    Memory --> Index["chroma_db/wound_clinical_docs.json"]
    Memory --> Docs["clinical_docs/"]
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
    K --> L["Load condition policy and tool plan"]
    L --> M["Retrieve evidence chunks"]
    M --> N["Run required tools through AgentTools"]
    N --> O["Route severity model"]
    O --> P["Simulated clinician Q&A"]
    P --> Q["Verify draft staging against evidence"]
    Q --> R["Return final report, probabilities, trace, evidence count, artifact path"]
```

1. The API loads `WoundAnalysisPipeline` at startup.
2. The uploaded image is validated and normalized to RGB.
3. The condition classifier returns top-k condition probabilities and confidence metadata.
4. The static demo exposes the condition as a dropdown so a reviewer can override the model label before downstream routing.
5. DFU and pressure-injury routes continue to segmentation, depth, severity routing, and agent evaluation; unsupported conditions stop after classification.
6. Segmentation is shown as a light-blue overlay on the original wound image; the relative depth map is shown beside the depth legend.
7. The agent loads the condition policy, builds a tool plan, retrieves local reference chunks, runs tool-backed severity inference, simulates context-driven Q&A, and verifies the draft stage.
8. The final response includes staging, probabilities, a brief report, verifier result, trace metadata, retrieved evidence count, and local artifact path.

## Agent Boundary

The current agent layer is deliberately conservative:

- It can run without an LLM key.
- Tool execution remains deterministic and auditable.
- Clinical references are retrieved from local indexed documents.
- The simulated Q&A helper only answers from provided context.
- Conditions outside the currently implemented DFU/pressure-injury severity routes are not forced through unsupported staging tools.

OpenAI-backed brain and verifier paths are configurable through `.env`, but the core workflow does not depend on them for basic local operation.

## Model Artifacts

The repository tracks model-loading code and checkpoint placement docs, not the binary checkpoints. See [CHECKPOINTS.md](CHECKPOINTS.md) for the expected directory layout.
