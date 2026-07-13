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
  currentScreen: 1,
  file: null,
  originalUrl: null,
  analysis: null,
  maskEnabled: true,
  depthEnabled: true,
  customMaskEnabled: false,
  customMaskApplied: false,
  customMaskBlob: null,
  busy: false,
};

const screenNames = {
  1: "Upload & result",
  2: "Analysis review",
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

const screens = [...document.querySelectorAll(".screen")];
const fileInput = document.querySelector("#image-upload");
const analyzeButton = document.querySelector("#analyze-image");
const maskToggle = document.querySelector("#mask-enabled");
const depthToggle = document.querySelector("#depth-enabled");
const customMaskToggle = document.querySelector("#custom-mask-enabled");
const customMaskEditor = document.querySelector("#custom-mask-editor");
const resultContent = document.querySelector("#result-content");

function setScreen(number) {
  if (number < 1 || number > 2 || (number === 2 && !state.analysis)) return;
  state.currentScreen = number;

  screens.forEach((screen) => {
    screen.hidden = Number(screen.dataset.screen) !== number;
  });

  document.querySelectorAll(".step-dot").forEach((dot, index) => {
    const step = index + 1;
    dot.classList.toggle("is-done", step < number);
    dot.classList.toggle("is-active", step === number);
  });

  document.querySelector("#step-label").textContent =
    `Step ${number} of 2 — ${screenNames[number]}`;
  document.querySelector("#previous-screen").disabled = number === 1;
  document.querySelector("#next-screen").disabled =
    number === 2 || !state.analysis;
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function setApiStatus(status, label) {
  const statusElement = document.querySelector("#api-status");
  statusElement.dataset.status = status;
  statusElement.lastChild.textContent = ` ${label}`;
}

function setBusy(isBusy, label = "Analyzing image…") {
  state.busy = isBusy;
  analyzeButton.disabled = isBusy || !state.file;
  fileInput.disabled = isBusy;
  const progress = document.querySelector("#analysis-progress");
  progress.hidden = !isBusy;
  document.querySelector("#analysis-progress-text").textContent = label;
}

function showError(message) {
  const error = document.querySelector("#analysis-error");
  document.querySelector("#analysis-error-text").textContent = message;
  error.hidden = !message;
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

function previewDataUrl(preview) {
  if (!preview?.data) return null;
  return `data:${preview.mime_type || "image/png"};base64,${preview.data}`;
}

function confidenceLevel(probability) {
  if (probability >= 0.8) return "High";
  if (probability >= 0.6) return "Moderate";
  return "Low";
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

function renderAnalysis(payload, preserveInputChoices = false) {
  state.analysis = payload;
  const condition = payload.condition || {};
  const severity = payload.severity || {};
  const mask = payload.mask || {};
  const depth = payload.depth || {};
  const conditionLabel = prettyCondition(condition.top1_label || "Unknown condition");

  document.querySelector("#condition-name").textContent = conditionLabel;
  document.querySelector("#inspector-badge").textContent = conditionLabel;
  document.querySelector(".badge-warning")?.remove();

  document.querySelector("#result-image").src = state.originalUrl;
  document.querySelector("#analysis-original-image").src = state.originalUrl;

  const maskUrl = previewDataUrl(mask.mask_overlay_preview);
  const depthUrl = previewDataUrl(depth.depth_preview);
  const maskImage = document.querySelector("#analysis-mask-image");
  const depthImage = document.querySelector("#analysis-depth-image");
  maskImage.src = maskUrl || "";
  depthImage.src = depthUrl || "";
  document.querySelector("#hold-mask-overlay").disabled = !maskUrl;
  document.querySelector("#hold-depth-map").disabled = !depthUrl;
  document.querySelector("#mask-area").textContent = maskAreaText(mask);

  renderProbabilityList(
    document.querySelector("#condition-probabilities"),
    (condition.top3 || []).map((item) => ({
      label: prettyCondition(item.class_name),
      probability: item.probability,
    })),
    conditionLabel,
    { highlightFirst: true },
  );

  const severityCard = document.querySelector("#severity-probability-card");
  const unavailable = document.querySelector("#severity-unavailable");
  if (severity.severity_available) {
    const prediction = severity.severity_prediction || "Severity unavailable";
    const isPressureInjury = /pressure injury/i.test(conditionLabel);
    const displayedPrediction = isPressureInjury
      ? pressureInjuryStageLabels[prediction] || prediction
      : prediction;
    document.querySelector("#severity-label").textContent = displayedPrediction;
    document.querySelector("#severity-description").textContent =
      severityDescriptions[prediction] || "Model-generated severity classification";
    setConfidence(confidenceLevel(Number(severity.severity_confidence) || 0));
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
      prediction,
    );
    document.querySelector("#severity-probability-title").textContent =
      prediction.startsWith("Grade") ? "Grade probabilities" : "Stage probabilities";
    severityCard.hidden = false;
    unavailable.hidden = true;
  } else {
    document.querySelector("#severity-label").textContent = "Not available";
    document.querySelector("#severity-description").textContent =
      severity.message || "Severity model is not available for this condition.";
    setConfidence(condition.confidence_level || "Low");
    severityCard.hidden = true;
    document.querySelector("#severity-unavailable-text").textContent =
      severity.message || "Severity model for this condition is still under development.";
    unavailable.hidden = false;
  }

  const warnings = [
    ...(payload.validation?.warnings || []),
    ...(mask.warnings || []),
    ...(depth.warnings || []),
    severity.warning,
  ].filter(Boolean);
  const warningBox = document.querySelector("#validation-warning");
  document.querySelector("#validation-warning-text").textContent = warnings.join(" ");
  warningBox.hidden = warnings.length === 0;

  if (!preserveInputChoices) {
    state.maskEnabled = Boolean(mask.mask_available);
    state.depthEnabled = Boolean(depth.depth_available);
    maskToggle.checked = state.maskEnabled;
    depthToggle.checked = state.depthEnabled;
    maskToggle.disabled = !mask.mask_available;
    depthToggle.disabled = !depth.depth_available;
  }
  document.querySelector("#update-severity").disabled = !severity.severity_available;

  resultContent.hidden = false;
  document.querySelector("#next-screen").disabled = false;
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

async function postImage(endpoint, extraData = {}, extraFiles = {}) {
  const form = new FormData();
  form.append("file", state.file, state.file.name);
  Object.entries(extraData).forEach(([key, value]) => form.append(key, value));
  Object.entries(extraFiles).forEach(([key, value]) => {
    form.append(key, value, `${key}.png`);
  });
  return parseResponse(
    await fetch(`${API_BASE}${endpoint}`, {
      method: "POST",
      body: form,
    }),
  );
}

async function analyzeImage() {
  if (!state.file || state.busy) return;
  showError("");
  setBusy(true, "Running condition, segmentation, depth, and severity models…");

  try {
    const payload = await postImage("/analyze");
    renderAnalysis(payload);
    setApiStatus("online", "API connected");
  } catch (error) {
    showError(error.message || "Analysis failed.");
    setApiStatus("offline", "API unavailable");
  } finally {
    setBusy(false);
  }
}

async function updateSeverity() {
  if (!state.analysis || state.busy) return;
  const condition = state.analysis.condition;
  const maskStatus = state.customMaskApplied
    ? "custom"
    : state.maskEnabled
      ? "accepted"
      : "none";
  const depthStatus = state.depthEnabled ? "accepted" : "rejected";
  const button = document.querySelector("#update-severity");
  const status = document.querySelector("#severity-update-status");

  button.disabled = true;
  status.textContent = "Updating severity model…";
  try {
    const files = {};
    if (maskStatus === "custom" && state.customMaskBlob) {
      files.mask_file = state.customMaskBlob;
    }
    const severity = await postImage(
      "/predict-severity",
      {
        selected_condition: condition.top1_label,
        condition_source: "model_accepted",
        mask_status: maskStatus,
        depth_status: depthStatus,
      },
      files,
    );
    state.analysis.severity = severity;
    renderAnalysis(state.analysis, true);
    status.textContent = severity.severity_available
      ? `Updated using ${severity.input_channels_used.join(" + ")}.`
      : severity.message;
    setScreen(1);
  } catch (error) {
    status.textContent = `Could not update severity: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

function configureHoldPreview({
  buttonSelector,
  activeClass,
  textSelector,
  idleText,
  activeText,
  clearClasses,
}) {
  const button = document.querySelector(buttonSelector);
  const preview = document.querySelector("#analysis-preview");
  const text = document.querySelector(textSelector);
  let activePointerId = null;

  function setActive(isActive) {
    if (button.disabled) return;
    if (isActive) clearClasses.forEach((className) => preview.classList.remove(className));
    preview.classList.toggle(activeClass, isActive);
    button.setAttribute("aria-pressed", String(isActive));
    text.textContent = isActive ? activeText : idleText;
  }

  button.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    activePointerId = event.pointerId;
    setActive(true);
    try {
      button.setPointerCapture?.(event.pointerId);
    } catch {
      // Pointer capture is best-effort for synthetic and older touch events.
    }
  });

  function release(event) {
    if (activePointerId !== null && event.pointerId !== activePointerId) return;
    activePointerId = null;
    setActive(false);
  }

  button.addEventListener("pointerup", release);
  button.addEventListener("pointercancel", release);
  button.addEventListener("lostpointercapture", () => {
    activePointerId = null;
    setActive(false);
  });
  button.addEventListener("keydown", (event) => {
    if (![" ", "Enter"].includes(event.key)) return;
    event.preventDefault();
    setActive(true);
  });
  button.addEventListener("keyup", (event) => {
    if (![" ", "Enter"].includes(event.key)) return;
    event.preventDefault();
    setActive(false);
  });
  button.addEventListener("blur", () => setActive(false));
}

function drawUploadedImageOnCanvas() {
  if (!state.originalUrl) return;
  const canvas = document.querySelector("#mask-base-canvas");
  const drawCanvas = document.querySelector("#mask-draw-canvas");
  const image = new Image();
  image.onload = () => {
    const maxWidth = 600;
    const maxHeight = 380;
    const scale = Math.min(maxWidth / image.naturalWidth, maxHeight / image.naturalHeight);
    const width = Math.max(1, Math.round(image.naturalWidth * scale));
    const height = Math.max(1, Math.round(image.naturalHeight * scale));
    canvas.width = width;
    canvas.height = height;
    drawCanvas.width = width;
    drawCanvas.height = height;
    document.querySelector("#draw-canvas-wrap").style.aspectRatio = `${width} / ${height}`;
    canvas.getContext("2d").drawImage(image, 0, 0, width, height);
    drawCanvas.getContext("2d").clearRect(0, 0, width, height);
  };
  image.src = state.originalUrl;
}

function exportCustomMask() {
  const source = document.querySelector("#mask-draw-canvas");
  const sourceContext = source.getContext("2d");
  const pixels = sourceContext.getImageData(0, 0, source.width, source.height).data;
  const output = document.createElement("canvas");
  output.width = source.width;
  output.height = source.height;
  const outputContext = output.getContext("2d");
  const mask = outputContext.createImageData(output.width, output.height);
  for (let index = 0; index < pixels.length; index += 4) {
    const value = pixels[index + 3] > 0 ? 255 : 0;
    mask.data[index] = value;
    mask.data[index + 1] = value;
    mask.data[index + 2] = value;
    mask.data[index + 3] = 255;
  }
  outputContext.putImageData(mask, 0, 0);
  return new Promise((resolve) => output.toBlob(resolve, "image/png"));
}

function configureMaskDrawing() {
  const canvas = document.querySelector("#mask-draw-canvas");
  const ctx = canvas.getContext("2d");
  const brushButton = document.querySelector("#draw-brush");
  const eraserButton = document.querySelector("#draw-eraser");
  const sizeInput = document.querySelector("#brush-size");
  let drawing = false;
  let tool = "brush";
  let previousPoint = null;

  function setTool(value) {
    tool = value;
    brushButton.classList.toggle("is-active", value === "brush");
    eraserButton.classList.toggle("is-active", value === "eraser");
  }

  function pointFromEvent(event) {
    const rect = canvas.getBoundingClientRect();
    return {
      x: ((event.clientX - rect.left) / rect.width) * canvas.width,
      y: ((event.clientY - rect.top) / rect.height) * canvas.height,
    };
  }

  function drawSegment(from, to) {
    ctx.save();
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.lineWidth = Number(sizeInput.value);
    if (tool === "eraser") {
      ctx.globalCompositeOperation = "destination-out";
      ctx.strokeStyle = "#000000";
    } else {
      ctx.globalCompositeOperation = "source-over";
      ctx.strokeStyle = "rgba(55, 138, 221, 0.55)";
    }
    ctx.beginPath();
    ctx.moveTo(from.x, from.y);
    ctx.lineTo(to.x, to.y);
    ctx.stroke();
    ctx.restore();
  }

  canvas.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    drawing = true;
    previousPoint = pointFromEvent(event);
    drawSegment(previousPoint, previousPoint);
    canvas.setPointerCapture?.(event.pointerId);
  });
  canvas.addEventListener("pointermove", (event) => {
    if (!drawing) return;
    event.preventDefault();
    const nextPoint = pointFromEvent(event);
    drawSegment(previousPoint, nextPoint);
    previousPoint = nextPoint;
  });
  window.addEventListener("pointerup", () => {
    drawing = false;
    previousPoint = null;
  });

  brushButton.addEventListener("click", () => setTool("brush"));
  eraserButton.addEventListener("click", () => setTool("eraser"));
  document.querySelector("#clear-mask").addEventListener("click", () => {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    state.customMaskApplied = false;
    state.customMaskBlob = null;
    document.querySelector("#mask-apply-status").textContent = "";
  });
  document.querySelector("#apply-mask").addEventListener("click", async () => {
    state.customMaskBlob = await exportCustomMask();
    state.customMaskApplied = Boolean(state.customMaskBlob);
    document.querySelector("#mask-apply-status").textContent =
      "Custom mask ready. Select Update severity result to use it.";
  });
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
  state.file = file;
  state.analysis = null;
  state.customMaskApplied = false;
  state.customMaskBlob = null;
  if (state.originalUrl) URL.revokeObjectURL(state.originalUrl);
  state.originalUrl = URL.createObjectURL(file);
  document.querySelector("#selected-file-name").textContent = file.name;
  document.querySelector("#selected-file").hidden = false;
  analyzeButton.disabled = false;
  resultContent.hidden = true;
  document.querySelector("#next-screen").disabled = true;
  showError("");
  drawUploadedImageOnCanvas();
});

analyzeButton.addEventListener("click", analyzeImage);
document.querySelector("#inspect-reasoning").addEventListener("click", () => setScreen(2));
document.querySelector("#inspector-back").addEventListener("click", () => setScreen(1));
document.querySelector("#back-to-result").addEventListener("click", () => setScreen(1));
document.querySelector("#previous-screen").addEventListener("click", () => {
  setScreen(state.currentScreen - 1);
});
document.querySelector("#next-screen").addEventListener("click", () => {
  setScreen(state.currentScreen + 1);
});
document.querySelector("#update-severity").addEventListener("click", updateSeverity);

maskToggle.addEventListener("change", () => {
  state.maskEnabled = maskToggle.checked;
  if (!state.maskEnabled) {
    customMaskToggle.checked = false;
    state.customMaskEnabled = false;
    state.customMaskApplied = false;
    state.customMaskBlob = null;
    customMaskEditor.hidden = true;
  }
});
depthToggle.addEventListener("change", () => {
  state.depthEnabled = depthToggle.checked;
});
customMaskToggle.addEventListener("change", () => {
  state.customMaskEnabled = customMaskToggle.checked;
  customMaskEditor.hidden = !state.customMaskEnabled;
  if (state.customMaskEnabled) {
    maskToggle.checked = true;
    state.maskEnabled = true;
    drawUploadedImageOnCanvas();
  } else {
    state.customMaskApplied = false;
    state.customMaskBlob = null;
  }
});

configureHoldPreview({
  buttonSelector: "#hold-mask-overlay",
  activeClass: "is-showing-mask",
  textSelector: "#mask-preview-button-text",
  idleText: "Hold for segmentation",
  activeText: "Segmentation overlay",
  clearClasses: ["is-showing-depth"],
});
configureHoldPreview({
  buttonSelector: "#hold-depth-map",
  activeClass: "is-showing-depth",
  textSelector: "#depth-preview-button-text",
  idleText: "Hold for depth map",
  activeText: "Depth map",
  clearClasses: ["is-showing-mask"],
});
configureMaskDrawing();
setScreen(1);
checkApi();
