# Blinded Evaluation Architecture

The evaluation layer is a compiled LangGraph in `app/evaluation/graph.py`. It runs only after the existing deterministic wound pipeline has produced its final outputs.

```text
START
  -> prepare_evaluation_state
  -> initial_verbalizer
  -> blinded_orchestrator
       -> targeted_verbalizer -> blinded_orchestrator
       -> pubagent            -> blinded_orchestrator
       -> commit_independent_assessment
  -> reveal_model_output
  -> comparison_adjudication
  -> finalize_evaluation
  -> END
```

The Orchestrator returns a schema-enforced `EvaluationAction` with exactly one route: `ASK_VERBALIZER`, `ASK_PUBAGENT`, or `COMMIT_ASSESSMENT`. The loop is capped by `MAX_EVALUATION_ITERATIONS = 5`; a fifth request for more evidence is converted into a best-available commit with unresolved uncertainty preserved.

## Blinding

`prepare_evaluation_state` separates wound evidence from `HiddenModelOutput`. Verbalizer and Orchestrator implementations receive dedicated Pydantic contexts containing no model condition, stage, confidence, probabilities, or classification fields. The commit node stores and hashes the independent assessment. The reveal and adjudication nodes reject execution if the assessment is absent or its digest changed.

## PubAgent

`PubAgentAdapter` imports the existing local `research_agent.main.build_agent` factory and calls the returned public `ResearchAgent.run(question, ...)` method. It does not implement search or ranking. PubAgent summaries, ranked quotes, sufficiency, and citation metadata are stored in graph state and the audit log.

## Invocation

`POST /agent/analyze` remains the integrated entry point. `WoundMindAgent` runs the existing deterministic tools, extracts blinded visual evidence, and calls:

```python
result = await EvaluationWorkflow(config).run(
    case_id=case_id,
    pipeline_output=completed_pipeline_output,
    case_evidence=allowed_wound_evidence,
)
```

The structured result is exposed under `output.evaluation_summary` and persisted as `case_artifacts/<case_id>/evaluation_output.json`. Model confidence, independent confidence, and final disposition confidence remain separate fields.
