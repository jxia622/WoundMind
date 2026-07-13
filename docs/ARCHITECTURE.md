# Architecture

WoundMind is organized as a diagnostic assistant around a deterministic tool pipeline. The agent layer chooses and explains tool use, while the model wrappers keep image inference isolated from API and UI code.

## System Components

```mermaid
flowchart TB
    UI["Streamlit / static demo"] --> API["FastAPI service"]
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

1. The API loads `WoundAnalysisPipeline` at startup.
2. The uploaded image is validated and normalized to RGB.
3. The condition classifier returns top-k condition probabilities and confidence metadata.
4. The agent loads the condition policy and builds a tool plan.
5. Segmentation and depth tools run only when the policy and route require them.
6. Severity routing selects the DFU or pressure-injury model variant based on accepted inputs.
7. Clinical memory retrieves local reference chunks for staging or severity verification.
8. The verifier formats the assessment with warnings, trace metadata, and the research-prototype disclaimer.

## Agent Boundary

The current agent layer is deliberately conservative:

- It can run without an LLM key.
- Tool execution remains deterministic and auditable.
- Clinical references are retrieved from local indexed documents.
- The simulated Q&A helper only answers from provided context.

OpenAI-backed brain and verifier paths are configurable through `.env`, but the core workflow does not depend on them for basic local operation.

## Model Artifacts

The repository tracks model-loading code and checkpoint placement docs, not the binary checkpoints. See [CHECKPOINTS.md](CHECKPOINTS.md) for the expected directory layout.
