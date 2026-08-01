const urlParams = new URLSearchParams(window.location.search);
const localApiHost = ["localhost", "127.0.0.1"].includes(window.location.hostname)
  ? "127.0.0.1"
  : window.location.hostname;
const API_BASE =
  urlParams.get("api") ||
  `${window.location.protocol === "file:" ? "http:" : window.location.protocol}//${
    localApiHost || "127.0.0.1"
  }:8000`;

const state = {
  file: null,
  originalUrl: null,
  validation: null,
  condition: null,
  selectedCondition: null,
  mask: null,
  depth: null,
  severity: null,
  agent: null,
  busy: false,
  evaluating: false,
  evaluationTimer: null,
  maskTintToken: 0,
};

const severityDescriptions = {
  "Stage 0": "Deep Tissue Injury",
  "Stage 1": "Non-blanchable erythema of intact skin",
  "Stage 2": "Partial-thickness skin loss with exposed dermis",
  "Stage 3": "Full-thickness skin loss",
  "Stage 4": "Full-thickness skin and tissue loss",
  "Stage 5": "Unstageable",
  "Grade 0": "No active diabetic foot ulcer grade detected",
  "Grade 1": "Superficial diabetic foot ulcer",
  "Grade 2": "Deeper ulcer involving tendon or capsule",
  "Grade 3": "Deep ulcer with abscess or bone involvement",
  "Grade 4": "Localized gangrene",
};

const pressureInjuryStageOrder = [
  "Stage 0",
  "Stage 1",
  "Stage 2",
  "Stage 3",
  "Stage 4",
  "Stage 5",
];

const pressureInjuryStageLabels = {
  "Stage 0": "Deep Tissue Injury",
  "Stage 1": "Stage 1",
  "Stage 2": "Stage 2",
  "Stage 3": "Stage 3",
  "Stage 4": "Stage 4",
  "Stage 5": "Unstageable",
};

const dfuGradeOrder = ["Grade 0", "Grade 1", "Grade 2", "Grade 3", "Grade 4"];

const conditionOptions = [
  "Healthy_Normal_Skin",
  "Pressure_Injury_(PI)",
  "Diabetic_Foot_Ulcers_(DFU)",
  "Venous_Leg_Ulcers_(VLU)",
  "Atopic_Dermatitis_(Eczema)",
  "Contact_Dermatitis",
  "Seborrheic_Dermatitis",
  "Acne",
  "Hidradenitis_Suppurativa_(HS)",
  "Sunburn",
  "Psoriasis",
  "Bruising-Contusions",
  "Melasma",
  "Cold_Injury",
  "Rosacea",
  "Surgical_Wounds",
  "Radiation_Injury",
  "Burn_Wounds",
];

const workflowSteps = {
  verify: "Verify image quality",
  classify: "Classify wound condition",
  route: "Check supported severity route",
  segment: "Generate wound segmentation",
  depth: "Estimate relative depth",
};

const evaluationTraceMessages = [
  "Preparing blinded wound evidence...",
  "Creating the initial grounded wound description...",
  "Running the blinded orchestrator...",
  "Requesting targeted wound or PubAgent evidence...",
  "Committing the independent assessment...",
  "Revealing the deterministic model output...",
  "Comparing the locked assessments...",
  "Preparing the auditable evaluation result...",
];

const fileInput = document.querySelector("#image-upload");
const analyzeButton = document.querySelector("#analyze-image");
const continueEvaluationButton = document.querySelector("#continue-evaluation");
const conditionSelect = document.querySelector("#condition-select");

function setApiStatus(status, label) {
  const statusElement = document.querySelector("#api-status");
  statusElement.dataset.status = status;
  statusElement.lastChild.textContent = ` ${label}`;
}

function setBusy(isBusy) {
  state.busy = isBusy;
  analyzeButton.disabled = isBusy || !state.file;
  fileInput.disabled = isBusy || state.evaluating;
}

function setEvaluating(isEvaluating) {
  state.evaluating = isEvaluating;
  continueEvaluationButton.disabled = isEvaluating;
  analyzeButton.disabled = isEvaluating || state.busy || !state.file;
  document.querySelector("#evaluation-spinner").hidden = !isEvaluating;
}

function showError(message) {
  const error = document.querySelector("#analysis-error");
  document.querySelector("#analysis-error-text").textContent = message;
  error.hidden = !message;
}

function showSection(selector, visible = true) {
  document.querySelector(selector).hidden = !visible;
}

function setActiveTrace(text, spinning = true) {
  showSection("#trace-card", true);
  document.querySelector("#active-trace-text").textContent = text;
  document.querySelector("#workflow-spinner").hidden = !spinning;
}

function addTimelineStep(id, label, status = "running", detail = "") {
  const list = document.querySelector("#workflow-timeline");
  let item = list.querySelector(`[data-step="${id}"]`);
  if (!item) {
    item = document.createElement("li");
    item.dataset.step = id;
    item.innerHTML = `
      <span class="timeline-icon"></span>
      <span class="timeline-copy">
        <strong></strong>
        <small></small>
      </span>
    `;
    list.append(item);
  }
  item.dataset.status = status;
  item.querySelector("strong").textContent = label;
  item.querySelector("small").textContent = detail;
}

function prettyCondition(value = "") {
  return value
    .replaceAll("_", " ")
    .replace(/\s+\(/g, " (")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .replace(/\bDfu\b/g, "DFU")
    .replace(/\bPi\b/g, "PI")
    .replace(/\bVlu\b/g, "VLU");
}

function isSupportedSeverityCondition(condition = "") {
  const normalized = condition.toLowerCase();
  return (
    normalized.includes("diabetic_foot") ||
    normalized.includes("dfu") ||
    normalized.includes("pressure_injury") ||
    normalized.includes("pressure injury")
  );
}

function selectedCondition() {
  return state.selectedCondition || state.condition?.top1_label || "";
}

function previewDataUrl(preview) {
  if (!preview?.data) return null;
  return `data:${preview.mime_type || "image/png"};base64,${preview.data}`;
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = reject;
    image.src = src;
  });
}

async function createTintedMaskDataUrl(maskUrl) {
  const image = await loadImage(maskUrl);
  const canvas = document.createElement("canvas");
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  const context = canvas.getContext("2d", { willReadFrequently: true });
  context.drawImage(image, 0, 0, canvas.width, canvas.height);

  const imageData = context.getImageData(0, 0, canvas.width, canvas.height);
  const { data } = imageData;
  for (let index = 0; index < data.length; index += 4) {
    const intensity = Math.max(data[index], data[index + 1], data[index + 2]);
    const alpha = intensity > 12 ? Math.max(72, Math.round((intensity / 255) * 154)) : 0;
    data[index] = 72;
    data[index + 1] = 190;
    data[index + 2] = 255;
    data[index + 3] = alpha;
  }
  context.putImageData(imageData, 0, 0);
  return canvas.toDataURL("image/png");
}

function confidenceLevel(probability) {
  if (probability >= 0.8) return "High";
  if (probability >= 0.6) return "Moderate";
  return "Low";
}

function formatPercent(probability) {
  if (probability == null || Number.isNaN(Number(probability))) return "-";
  return `${Math.round(Number(probability) * 100)}%`;
}

function setConfidence(level) {
  const element = document.querySelector("#confidence-level");
  const normalized = (level || "Low").toLowerCase();
  element.textContent = level || "Low";
  element.className = `confidence-level confidence-${normalized}`;
}

function renderProbabilityList(
  container,
  entries,
  activeLabel,
  { highlightFirst = false } = {},
) {
  container.replaceChildren();
  entries.forEach((entry, index) => {
    const label = entry.label;
    const probability = Math.max(0, Math.min(1, Number(entry.probability) || 0));
    const row = document.createElement("div");
    row.className = "probability-row";
    if (container.id === "condition-probabilities") {
      row.classList.add("condition-probability-row");
    }
    if (entry.key === activeLabel || label === activeLabel || (highlightFirst && index === 0)) {
      row.classList.add("is-active");
    }

    const labelElement = document.createElement("span");
    labelElement.className = "probability-label";
    labelElement.textContent = label;

    const track = document.createElement("span");
    track.className = "probability-track";
    const fill = document.createElement("span");
    fill.className = `probability-fill ${
      index === 0
        ? "probability-active"
        : index === 1
          ? "probability-low"
          : "probability-high"
    }`;
    fill.style.width = `${Math.max(probability * 100, probability > 0 ? 1 : 0)}%`;
    track.append(fill);

    const value = document.createElement("span");
    value.className = "probability-value";
    value.textContent = `${Math.round(probability * 100)}%`;

    row.append(labelElement, track, value);
    container.append(row);
  });
}

function maskAreaText(mask) {
  if (!mask?.mask_area_pixels) return "Not available";
  return `${Number(mask.mask_area_pixels).toLocaleString()} px²`;
}

async function parseResponse(response) {
  let payload;
  try {
    payload = await response.json();
  } catch {
    payload = { detail: await response.text() };
  }
  if (!response.ok) {
    throw new Error(
      typeof payload.detail === "string"
        ? payload.detail
        : JSON.stringify(payload.detail || payload),
    );
  }
  return payload;
}

async function postImage(endpoint, fieldName = "file", extraData = {}) {
  const form = new FormData();
  form.append(fieldName, state.file, state.file.name);
  Object.entries(extraData).forEach(([key, value]) => form.append(key, value));
  return parseResponse(
    await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      body: form,
    }),
  );
}

function resetWorkflowOutput() {
  state.validation = null;
  state.condition = null;
  state.selectedCondition = null;
  state.mask = null;
  state.depth = null;
  state.severity = null;
  state.agent = null;
  window.clearInterval(state.evaluationTimer);
  state.evaluationTimer = null;

  [
    "#trace-card",
    "#condition-card",
    "#review-card",
    "#evaluation-card",
    "#final-result",
    "#condition-stop",
    "#validation-warning",
    "#verifier-citation",
    "#verifier-flags",
  ].forEach((selector) => showSection(selector, false));
  document.querySelector("#workflow-timeline").replaceChildren();
  document.querySelector("#agent-trace-list").replaceChildren();
  showError("");
}

function renderValidation(validation) {
  const warnings = validation.warnings || [];
  document.querySelector("#validation-warning-text").textContent = warnings.join(" ");
  showSection("#validation-warning", warnings.length > 0);
}

function renderCondition(condition) {
  state.selectedCondition = state.selectedCondition || condition.top1_label;
  const selected = selectedCondition();
  const modelCondition = condition.top1_label;
  showSection("#condition-card", true);
  conditionSelect.value = selected;
  const overrideText =
    selected === modelCondition
      ? `Model prediction confidence ${formatPercent(condition.confidence)} - ${condition.confidence_level || confidenceLevel(condition.confidence)}`
      : `User override from ${prettyCondition(modelCondition)} (${formatPercent(condition.confidence)} model confidence)`;
  document.querySelector("#condition-confidence").textContent = overrideText;
  renderProbabilityList(
    document.querySelector("#condition-probabilities"),
    (condition.top3 || []).map((item) => ({
      label: prettyCondition(item.class_name),
      probability: item.probability,
    })),
    prettyCondition(modelCondition),
    { highlightFirst: true },
  );
}

async function renderReview() {
  const maskUrl = previewDataUrl(state.mask?.mask_preview);
  const depthUrl = previewDataUrl(state.depth?.depth_preview);
  const maskToken = (state.maskTintToken += 1);
  document.querySelector("#analysis-mask-original-image").src = state.originalUrl || "";
  document.querySelector("#analysis-mask-tint-image").src = "";
  document.querySelector("#analysis-depth-image").src = depthUrl || "";
  document.querySelector("#mask-area").textContent = maskAreaText(state.mask);
  showSection("#review-card", true);
  if (!maskUrl) return;

  try {
    const tintedMaskUrl = await createTintedMaskDataUrl(maskUrl);
    if (maskToken === state.maskTintToken) {
      document.querySelector("#analysis-mask-tint-image").src = tintedMaskUrl;
    }
  } catch {
    if (maskToken === state.maskTintToken) {
      document.querySelector("#analysis-mask-tint-image").src =
        previewDataUrl(state.mask?.mask_overlay_preview) || "";
    }
  }
}

function renderSeverityProbabilities(severity, conditionLabel) {
  const isPressureInjury = /pressure injury/i.test(conditionLabel);
  const classProbabilities = severity.class_probabilities || {};
  const classOrder = isPressureInjury ? pressureInjuryStageOrder : dfuGradeOrder;
  const probabilities = classOrder
    .filter((key) => Object.hasOwn(classProbabilities, key))
    .map((key) => ({
      key,
      label: isPressureInjury ? pressureInjuryStageLabels[key] : key,
      probability: classProbabilities[key],
    }));
  renderProbabilityList(
    document.querySelector("#severity-probabilities"),
    probabilities,
    severity.severity_prediction,
  );
  document.querySelector("#severity-probability-title").textContent =
    (severity.severity_prediction || "").startsWith("Grade")
      ? "Grade probabilities"
      : "Stage probabilities";
}

function traceOutputLine(step) {
  const output = step.outputs_summary || {};
  const input = step.inputs_summary || {};
  if (step.tool === "validate_image") {
    return output.passed ? "Image quality accepted" : "Quality warnings recorded";
  }
  if (step.tool === "classify_condition") {
    return `${prettyCondition(output.top1_label)} (${formatPercent(output.confidence)})`;
  }
  if (step.tool === "override_condition") {
    return `User selected ${prettyCondition(output.selected_condition)} over ${prettyCondition(input.model_top1)}`;
  }
  if (step.tool === "load_condition_policy") {
    const tools = output.runnable_tools || [];
    const toolCount = Array.isArray(tools) ? tools.length : 0;
    return `${toolCount} implemented runtime tools selected`;
  }
  if (step.tool === "select_mask") {
    return output.rationale || "Mask selected";
  }
  if (step.tool === "predict_severity") {
    return `${output.severity_prediction || "No severity"} via ${output.model_used || "route"}`;
  }
  if (step.tool === "verify_assessment") {
    return output.result || output.flag || "Verification complete";
  }
  if (step.tool === "extract_visual_evidence") {
    return output.vlm_used
      ? `${(output.findings || []).length} visual findings extracted`
      : (output.limitations || ["VLM unavailable"])[0];
  }
  if (step.tool === "prepare_evaluation_state") {
    return output.blinding_active
      ? `${(output.case_evidence_fields || []).length} blinded evidence groups prepared`
      : "Evaluation state prepared";
  }
  if (step.tool === "initial_verbalizer") {
    return output.description || "Initial wound description complete";
  }
  if (step.tool === "blinded_orchestrator") {
    return output.rationale || input.route || "Blinded action selected";
  }
  if (step.tool === "targeted_verbalizer") {
    return output.answer || output.unavailable_reason || "Targeted wound answer recorded";
  }
  if (step.tool === "pubagent") {
    return output.summary || output.error || "Clinical evidence recorded";
  }
  if (step.tool === "commit_independent_assessment") {
    const assessment = output.assessment || {};
    return `${assessment.condition || "Unresolved condition"}, ${assessment.stage || "unresolved stage"} locked`;
  }
  if (step.tool === "reveal_model_output") {
    return output.revealed_after_commit
      ? "Deterministic result revealed after independent commit"
      : "Model result revealed";
  }
  if (step.tool === "comparison_adjudication") {
    return output.final_evaluation?.status || "Assessment comparison complete";
  }
  if (step.tool === "finalize_evaluation") {
    return output.status || "Evaluation finalized";
  }
  if (step.tool === "diagnostic_agent") {
    return `${output.severity_stage || "No stage"} proposal, ${(output.uncertainties || []).length} uncertainties`;
  }
  if (step.tool === "evaluator") {
    return output.reasoning || output.flag || output.result || "Evaluator complete";
  }
  if (step.tool === "retrieve_docs") {
    return `${(output.chunks || []).length} evidence chunks`;
  }
  if (step.tool === "segment_wound") {
    return output.mask_available ? "Mask generated" : "Mask unavailable";
  }
  if (step.tool === "depth_map") {
    return output.depth_available ? "Depth map generated" : "Depth unavailable";
  }
  if (step.tool === "ask_clinician") {
    return `${input.question || "Question"} ${output.answer ? `- ${output.answer}` : ""}`;
  }
  if (step.tool === "skip_tool") {
    return output.reason || "Tool skipped by policy";
  }
  return Object.keys(output).length ? "Complete" : "";
}

function renderAgentTrace(trace = []) {
  const list = document.querySelector("#agent-trace-list");
  list.replaceChildren();
  trace.forEach((step) => {
    const item = document.createElement("li");
    item.dataset.status = "done";
    item.innerHTML = `
      <span class="timeline-icon"></span>
      <span class="timeline-copy">
        <strong></strong>
        <small></small>
      </span>
    `;
    item.querySelector("strong").textContent = step.tool.replaceAll("_", " ");
    item.querySelector("small").textContent = traceOutputLine(step);
    list.append(item);
  });
}

function briefReport(output, severity) {
  const stage = output.severity_stage || severity?.severity_prediction || "no available stage";
  const condition = prettyCondition(output.condition || selectedCondition() || "condition");
  const confidence = output.severity_confidence ?? severity?.severity_confidence;
  const modelVariant = output.model_variant_used || severity?.model_used;
  const modelLine = modelVariant ? ` using ${modelVariant}` : "";
  const evaluation = output.evaluation_summary || {};
  const independent = evaluation.independent_evaluation || {};
  const finalEvaluation = evaluation.final_evaluation || {};
  const independentAssessment = independent.condition
    ? `${prettyCondition(independent.condition)}${independent.stage ? `, ${independent.stage}` : ""}`
    : "insufficient evidence for an independent condition";
  const disposition = finalEvaluation.status || output.verifier_result || "INSUFFICIENT_EVIDENCE";
  return `The deterministic pipeline returned ${condition}, ${stage} at ${formatPercent(confidence)} confidence${modelLine}. The blinded evaluator committed ${independentAssessment} before reveal, then returned ${disposition}.`;
}

function renderFinalResult(agentPayload) {
  const output = agentPayload.output || {};
  const severity = state.severity || {};
  const conditionLabel = prettyCondition(output.condition || selectedCondition() || "Condition");
  const stage = output.severity_stage || severity.severity_prediction || "Not verified";
  const evaluation = output.evaluation_summary || {};
  const finalEvaluation = evaluation.final_evaluation || {};
  const displayedStage = /pressure injury/i.test(conditionLabel)
    ? pressureInjuryStageLabels[stage] || stage
    : stage;

  document.querySelector("#final-condition-label").textContent = conditionLabel;
  document.querySelector("#severity-label").textContent = displayedStage;
  document.querySelector("#severity-description").textContent =
    severityDescriptions[stage] || "Deterministic model output available for review.";
  setConfidence(confidenceLevel(output.severity_confidence ?? severity.severity_confidence ?? 0));
  renderSeverityProbabilities(severity, conditionLabel);

  document.querySelector("#brief-report").textContent = briefReport(output, severity);
  document.querySelector("#agent-case-id").textContent = agentPayload.case_id || output.case_id || "-";
  document.querySelector("#verifier-result").textContent =
    finalEvaluation.status || output.verifier_result || "-";
  document.querySelector("#model-variant").textContent =
    output.model_variant_used || severity.model_used || "-";
  document.querySelector("#agent-trace-count").textContent =
    String(output.agent_trace?.length || 0);
  const pubAgentEvidenceCount = (evaluation.pubagent_history || []).reduce(
    (count, response) => count + (response.evidence?.length || 0),
    0,
  );
  document.querySelector("#agent-evidence-count").textContent = String(pubAgentEvidenceCount);
  document.querySelector("#agent-artifact-path").textContent = output.artifact_path || "-";

  document.querySelector("#verifier-citation-text").textContent = output.citation || "";
  showSection("#verifier-citation", Boolean(output.citation));
  const flags = output.flags || [];
  document.querySelector("#verifier-flags-text").textContent = flags.join(" ");
  showSection("#verifier-flags", flags.length > 0);
  showSection("#final-result", true);
}

async function runInitialAnalysis() {
  if (!state.file || state.busy) return;
  resetWorkflowOutput();
  setBusy(true);
  showError("");

  try {
    setActiveTrace("Verifying image quality...");
    addTimelineStep("verify", workflowSteps.verify, "running", "Checking dimensions, format, and blur score");
    state.validation = await postImage("/validate-image");
    renderValidation(state.validation);
    addTimelineStep(
      "verify",
      workflowSteps.verify,
      state.validation.passed ? "done" : "warning",
      state.validation.passed ? "Image quality passed" : "Warnings found",
    );

    setActiveTrace("Classifying wound condition...");
    addTimelineStep("classify", workflowSteps.classify, "running", "Running condition classifier");
    state.condition = await postImage("/predict-condition");
    state.selectedCondition = state.condition.top1_label;
    renderCondition(state.condition);
    addTimelineStep(
      "classify",
      workflowSteps.classify,
      "done",
      `${prettyCondition(state.condition.top1_label)} - ${formatPercent(state.condition.confidence)}`,
    );

    await runDownstreamForSelectedCondition();
    setApiStatus("online", "API connected");
  } catch (error) {
    showError(error.message || "Analysis failed.");
    setApiStatus("offline", "API unavailable");
    setActiveTrace("Workflow stopped because a request failed.", false);
  } finally {
    setBusy(false);
  }
}

async function runDownstreamForSelectedCondition() {
  const condition = selectedCondition();
  showSection("#review-card", false);
  showSection("#evaluation-card", false);
  showSection("#final-result", false);
  showSection("#condition-stop", false);
  ["route", "segment", "depth"].forEach((id) => {
    document.querySelector(`#workflow-timeline [data-step="${id}"]`)?.remove();
  });

  setActiveTrace("Checking whether downstream severity workflow is available...");
  addTimelineStep("route", workflowSteps.route, "running", "DFU and pressure injury are currently supported");
  if (!isSupportedSeverityCondition(condition)) {
    addTimelineStep("route", workflowSteps.route, "warning", "Workflow stops after classification");
    showSection("#condition-stop", true);
    setActiveTrace("Classification complete. Severity workflow is not built for this condition yet.", false);
    return;
  }
  addTimelineStep("route", workflowSteps.route, "done", "Severity workflow available");

  if (!state.mask) {
    setActiveTrace("Generating wound segmentation...");
    addTimelineStep("segment", workflowSteps.segment, "running", "Running U-Net++ mask model");
    state.mask = await postImage("/generate-mask", "file", { case_id: state.condition.case_id });
    addTimelineStep(
      "segment",
      workflowSteps.segment,
      state.mask.mask_available ? "done" : "warning",
      state.mask.mask_available ? "Mask overlay ready" : "Mask unavailable",
    );
  } else {
    addTimelineStep("segment", workflowSteps.segment, "done", "Existing mask overlay ready");
  }

  if (!state.depth) {
    setActiveTrace("Estimating relative depth map...");
    addTimelineStep("depth", workflowSteps.depth, "running", "Running Depth Anything V2");
    state.depth = await postImage("/generate-depth", "file", { case_id: state.condition.case_id });
    addTimelineStep(
      "depth",
      workflowSteps.depth,
      state.depth.depth_available ? "done" : "warning",
      state.depth.depth_available ? "Depth preview ready" : "Depth unavailable",
    );
  } else {
    addTimelineStep("depth", workflowSteps.depth, "done", "Existing depth preview ready");
  }

  state.severity = await postImage("/predict-severity", "file", {
    selected_condition: condition,
    condition_source: condition === state.condition.top1_label ? "model_accepted" : "user_override",
    mask_status: state.mask.mask_available ? "accepted" : "none",
    depth_status: state.depth.depth_available ? "accepted" : "rejected",
  });

  await renderReview();
  setActiveTrace("Visual review ready. Continue to evaluation when ready.", false);
}

function startEvaluationTrace() {
  let index = 0;
  showSection("#evaluation-card", true);
  document.querySelector("#evaluation-trace-text").textContent = evaluationTraceMessages[0];
  document.querySelector("#agent-trace-list").replaceChildren();
  state.evaluationTimer = window.setInterval(() => {
    index = Math.min(index + 1, evaluationTraceMessages.length - 1);
    document.querySelector("#evaluation-trace-text").textContent =
      evaluationTraceMessages[index];
  }, 1300);
}

async function continueToEvaluation() {
  if (!state.file || state.evaluating) return;
  setEvaluating(true);
  startEvaluationTrace();
  showSection("#final-result", false);

  try {
    state.agent = await postImage("/agent/analyze", "image", {
      case_id: state.condition?.case_id || "",
      selected_condition: selectedCondition(),
    });
    window.clearInterval(state.evaluationTimer);
    state.evaluationTimer = null;
    document.querySelector("#evaluation-trace-text").textContent =
      "Blinded evaluation complete.";
    renderAgentTrace(state.agent.output?.agent_trace || []);
    renderFinalResult(state.agent);
    setApiStatus("online", "API connected");
  } catch (error) {
    showError(error.message || "Evaluation failed.");
    document.querySelector("#evaluation-trace-text").textContent =
      "Evaluation failed.";
    setApiStatus("offline", "API unavailable");
  } finally {
    setEvaluating(false);
  }
}

async function checkApi() {
  try {
    const response = await fetch(`${API_BASE}/health`);
    if (!response.ok) throw new Error();
    setApiStatus("online", "API connected");
  } catch {
    setApiStatus("offline", "Start FastAPI on port 8000");
  }
}

fileInput.addEventListener("change", () => {
  const [file] = fileInput.files;
  if (!file) return;
  resetWorkflowOutput();
  state.file = file;
  if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
  state.originalUrl = URL.createObjectURL(file);
  document.querySelector("#selected-file-name").textContent = file.name;
  document.querySelector("#selected-file").hidden = false;
  analyzeButton.disabled = false;
  showError("");
});

analyzeButton.addEventListener("click", runInitialAnalysis);
continueEvaluationButton.addEventListener("click", continueToEvaluation);
conditionSelect.addEventListener("change", async () => {
  if (!state.condition || state.busy || state.evaluating) return;
  state.selectedCondition = conditionSelect.value;
  renderCondition(state.condition);
  try {
    setBusy(true);
    await runDownstreamForSelectedCondition();
  } catch (error) {
    showError(error.message || "Condition override failed.");
  } finally {
    setBusy(false);
  }
});

conditionOptions.forEach((condition) => {
  const option = document.createElement("option");
  option.value = condition;
  option.textContent = prettyCondition(condition);
  conditionSelect.append(option);
});

checkApi();
