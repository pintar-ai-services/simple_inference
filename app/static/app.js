const POLL_INTERVAL_MS = 1000;
const POLL_TIMEOUT_MS = 10 * 60 * 1000;
let previewImage = null;

function activateTab(name) { document.querySelectorAll("[role=tab]").forEach(tab => { const selected = tab.dataset.tab === name; tab.setAttribute("aria-selected", String(selected)); tab.tabIndex = selected ? 0 : -1; }); document.querySelectorAll(".workflow-panel").forEach(panel => { panel.hidden = panel.id !== `panel-${name}`; }); }
document.querySelectorAll("[role=tab]").forEach((tab, index, tabs) => { tab.addEventListener("click", () => activateTab(tab.dataset.tab)); tab.addEventListener("keydown", event => { if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key)) return; event.preventDefault(); const next = event.key === "Home" ? 0 : event.key === "End" ? tabs.length - 1 : (index + (event.key === "ArrowRight" ? 1 : -1) + tabs.length) % tabs.length; tabs[next].focus(); activateTab(tabs[next].dataset.tab); }); });
function status(id, message) { document.querySelector(id).textContent = message; }
function fileLabel(input) { const label = input.closest("label").querySelector(".file-name"); if (input.files[0]) label.textContent = `${input.files[0].name} · ${(input.files[0].size / 1024 / 1024).toFixed(2)} MB`; }
async function previewDetectionImage(file) { if (!file) return; const url = URL.createObjectURL(file); try { previewImage = new Image(); previewImage.src = url; await previewImage.decode(); const canvas = document.querySelector("#detection-canvas"), context = canvas.getContext("2d"); canvas.width = previewImage.naturalWidth; canvas.height = previewImage.naturalHeight; context.drawImage(previewImage, 0, 0); canvas.hidden = false; canvas.closest(".detection-dropzone").classList.add("has-preview"); document.querySelector("#detection-output").hidden = true; } finally { URL.revokeObjectURL(url); } }
async function previewClassificationImage(file) { if (!file) return; const url = URL.createObjectURL(file); try { const image = new Image(); image.src = url; await image.decode(); const canvas = document.querySelector("#classification-canvas"), context = canvas.getContext("2d"); canvas.width = image.naturalWidth; canvas.height = image.naturalHeight; context.drawImage(image, 0, 0); canvas.hidden = false; canvas.closest(".classification-dropzone").classList.add("has-preview"); document.querySelector("#classification-output").hidden = true; } finally { URL.revokeObjectURL(url); } }
async function pollJob(jobId, statusId) { const deadline = Date.now() + POLL_TIMEOUT_MS; for (;;) { const response = await fetch(`/jobs/${jobId}`), job = await response.json(); if (!response.ok) throw new Error(job.detail || "Could not read job status."); status(statusId, `Job ${jobId}: ${job.status}`); if (job.status === "finished") return job.result; if (["failed", "stopped", "canceled"].includes(job.status)) throw new Error(job.error || `Job ended with status: ${job.status}.`); if (Date.now() >= deadline) throw new Error("Timed out waiting for the job."); await new Promise(resolve => setTimeout(resolve, POLL_INTERVAL_MS)); } }
function drawDetections(detections) { const canvas = document.querySelector("#detection-canvas"), context = canvas.getContext("2d"); canvas.width = previewImage.naturalWidth; canvas.height = previewImage.naturalHeight; context.drawImage(previewImage, 0, 0); context.lineWidth = Math.max(2, canvas.width / 350); context.font = `${Math.max(14, canvas.width / 45)}px sans-serif`; context.textBaseline = "top"; const list = document.querySelector("#detections"); list.replaceChildren(); for (const item of detections) { const [x1, y1, x2, y2] = item.box, caption = `${item.label} ${(item.score * 100).toFixed(1)}%`; context.strokeStyle = "#e72d21"; context.strokeRect(x1, y1, x2 - x1, y2 - y1); const width = context.measureText(caption).width + 10, height = Math.max(20, canvas.width / 28); context.fillStyle = "#0067b5"; context.fillRect(x1, Math.max(0, y1 - height), width, height); context.fillStyle = "white"; context.fillText(caption, x1 + 5, Math.max(0, y1 - height + 2)); const li = document.createElement("li"); li.textContent = caption; list.append(li); } document.querySelector("#detection-count").textContent = `${detections.length} object${detections.length === 1 ? "" : "s"} detected`; document.querySelector("#detection-output").hidden = false; }
function bindImageForm(formId, inputId, endpoint, statusId, done) { const form = document.querySelector(formId), input = document.querySelector(inputId), button = form.querySelector("button"), preview = endpoint === "/jobs" ? previewDetectionImage : endpoint === "/classification/jobs" ? previewClassificationImage : null; input.addEventListener("change", async () => { fileLabel(input); if (preview) await preview(input.files[0]); }); form.addEventListener("submit", async event => { event.preventDefault(); const file = input.files[0]; if (!file) return; button.disabled = true; try { if (endpoint === "/jobs" && !previewImage) await previewDetectionImage(file); if (endpoint === "/classification/jobs" && document.querySelector("#classification-canvas").hidden) await previewClassificationImage(file); const body = new FormData(); body.append("image", file); status(statusId, "Uploading image and creating a job…"); const response = await fetch(endpoint, {method: "POST", body}), job = await response.json(); if (!response.ok) throw new Error(job.detail || "Could not create job."); const result = await pollJob(job.job_id, statusId); done(result); status(statusId, "Analysis complete."); } catch (error) { status(statusId, `Error: ${error.message}`); } finally { button.disabled = false; } }); }
bindImageForm("#detection-form", "#detection-image", "/jobs", "#detection-status", result => drawDetections(result.detections));
bindImageForm("#classification-form", "#classification-image", "/classification/jobs", "#classification-status", result => { const list = document.querySelector("#classification-predictions"); list.replaceChildren(); result.predictions.forEach(item => { const li = document.createElement("li"); li.textContent = `${item.label} — ${(item.score * 100).toFixed(1)}%`; list.append(li); }); document.querySelector("#classification-output").hidden = false; });
let annModels = [];
let annSelected = null;
let annImageFile = null;
let annRevision = 0;
const annForm = document.querySelector("#ann-form");
const annSelect = document.querySelector("#ann-model");
const annImage = document.querySelector("#ann-image");

function clearAnnResult() {
  annRevision += 1;
  document.querySelector("#ann-output").hidden = true;
  document.querySelector("#ann-processed").hidden = true;
}

function selectAnnModel() {
  clearAnnResult();
  annSelected = annModels.find(model => model.model_id === annSelect.value) || null;
  annImageFile = null;
  annImage.value = "";
  annImage.closest("label").querySelector(".file-name").textContent = "No file selected";
  document.querySelector("#ann-image-canvas").hidden = true;
  const fields = document.querySelector("#ann-fields");
  fields.replaceChildren();
  document.querySelector("#ann-submit").disabled = !annSelected;
  document.querySelector("#ann-example").disabled = !annSelected;
  document.querySelector("#ann-image-input").hidden = !annSelected || annSelected.input.type !== "image";
  if (!annSelected) {
    status("#ann-model-summary", "The default ANN is unavailable. Correct its deployment files and restart the API and worker, or choose another registered model.");
    status("#ann-status", "Select an available model to begin.");
    status("#ann-preprocessing", "Model information is unavailable.");
    return;
  }
  const model = annSelected;
  status("#ann-model-name", model.display_name);
  const accuracy = model.metrics?.test_accuracy;
  status("#ann-model-summary", `Version ${model.model_version} · ${model.output.task.replaceAll("_", " ")}${accuracy === undefined ? "" : ` · Held-out accuracy ${(accuracy * 100).toFixed(1)}%`}.`);
  if (model.input.type === "tabular") {
    model.input.features.forEach((name, index) => {
      const label = document.createElement("label");
      const input = document.createElement("input");
      input.id = `ann-feature-${index}`;
      input.type = "number";
      input.step = "any";
      input.required = true;
      input.dataset.feature = name;
      label.htmlFor = input.id;
      label.append(document.createTextNode(name), input);
      fields.append(label);
    });
    status("#ann-input-description", `Enter ${model.input.features.length} numeric values, or load the supplied example.`);
    status("#ann-preprocessing", model.input.preprocessing.type === "standard_scaler" ? "Values are placed in training order and scaled with the mean and scale saved by the training notebook. The model returns class scores, which are converted to probabilities." : "Values are placed in the feature order saved by the training notebook and passed to the model without scaling.");
  } else {
    const [height, width] = model.input.preprocessing.resize;
    status("#ann-input-description", `Upload a light digit on a dark background. The model uses a ${width} × ${height} grayscale image.`);
    status("#ann-preprocessing", `The server converts the image to grayscale, resizes it to ${width} × ${height}, ${model.input.preprocessing.invert ? "inverts its colors, " : ""}and scales pixel values by ${model.input.preprocessing.pixel_scale}. The network flattens the image and returns class probabilities.`);
  }
  status("#ann-status", "Ready. Load an example or enter your own input.");
}

async function showAnnImage(file, revision) {
  const url = URL.createObjectURL(file);
  try {
    const image = new Image();
    image.src = url;
    await image.decode();
    if (revision !== annRevision) return;
    const canvas = document.querySelector("#ann-image-canvas");
    canvas.width = image.naturalWidth;
    canvas.height = image.naturalHeight;
    canvas.getContext("2d").drawImage(image, 0, 0);
    canvas.hidden = false;
  } finally { URL.revokeObjectURL(url); }
}

annSelect.addEventListener("change", selectAnnModel);
annForm.addEventListener("input", event => { if (event.target !== annSelect && event.target !== annImage) clearAnnResult(); });
annImage.addEventListener("change", async () => {
  clearAnnResult();
  annImageFile = annImage.files[0] || null;
  document.querySelector("#ann-image-canvas").hidden = true;
  fileLabel(annImage);
  if (annImageFile) {
    try { await showAnnImage(annImageFile, annRevision); }
    catch { annImageFile = null; status("#ann-status", "Choose a valid PNG, JPEG, or WebP image."); }
  }
});

document.querySelector("#ann-example").addEventListener("click", async () => {
  if (!annSelected) return;
  clearAnnResult();
  const revision = annRevision;
  try {
    const example = annSelected.example;
    if (annSelected.input.type === "tabular") {
      document.querySelectorAll("#ann-fields input").forEach(input => { input.value = example.features[input.dataset.feature]; });
    } else {
      const {pixels, shape} = example.image;
      const [height, width] = shape;
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const context = canvas.getContext("2d");
      const data = context.createImageData(width, height);
      pixels.flat().forEach((value, i) => { data.data.set([value, value, value, 255], i * 4); });
      context.putImageData(data, 0, 0);
      const blob = await new Promise(resolve => canvas.toBlob(resolve, "image/png"));
      if (revision !== annRevision) return;
      annImageFile = new File([blob], "example.png", {type: "image/png"});
      annImage.value = "";
      annImage.closest("label").querySelector(".file-name").textContent = "Notebook example image";
      await showAnnImage(annImageFile, revision);
    }
    if (revision === annRevision) status("#ann-status", "Example loaded. Run prediction to try the model.");
  } catch { status("#ann-status", "The example is invalid. Export the model files again from the notebook."); }
});

annForm.addEventListener("submit", async event => {
  event.preventDefault();
  if (!annSelected) return;
  clearAnnResult();
  const revision = annRevision;
  const model = annSelected;
  const button = document.querySelector("#ann-submit");
  button.disabled = true;
  // Keep the selected model and inputs fixed while the queued request completes.
  annSelect.disabled = true;
  document.querySelector("#ann-example").disabled = true;
  annForm.querySelectorAll("input").forEach(input => { input.disabled = true; });
  try {
    let body, headers, suffix;
    if (model.input.type === "image") {
      if (!annImageFile) throw new Error("Choose an image or load the example first.");
      body = new FormData();
      body.append("image", annImageFile);
      suffix = "image-jobs";
    } else {
      const features = Object.fromEntries(Array.from(document.querySelectorAll("#ann-fields input"), input => [input.dataset.feature, Number(input.value)]));
      if (!Object.values(features).every(Number.isFinite)) throw new Error("Enter a finite number for every feature.");
      body = JSON.stringify({features});
      headers = {"content-type": "application/json"};
      suffix = "jobs";
    }
    status("#ann-status", "Submitting ANN inference…");
    const response = await fetch(`/ann/models/${encodeURIComponent(model.model_id)}/${suffix}`, {method: "POST", headers, body});
    const job = await response.json();
    if (!response.ok) throw new Error(job.detail || "Could not create the ANN job.");
    const result = await pollJob(job.job_id, "#ann-status");
    if (revision !== annRevision) return;
    status("#ann-prediction", String(result.prediction));
    status("#ann-result-model", `${result.model_id} · version ${result.model_version}`);
    const list = document.querySelector("#ann-probabilities");
    list.replaceChildren();
    Object.entries(result.probabilities || {}).forEach(([label, probability]) => {
      const item = document.createElement("li");
      item.textContent = `${label} — ${(probability * 100).toFixed(1)}%`;
      list.append(item);
    });
    if (result.image_preview) {
      document.querySelector("#ann-processed-image").src = result.image_preview.data_url;
      document.querySelector("#ann-processed").hidden = false;
    }
    document.querySelector("#ann-output").hidden = false;
    status("#ann-status", "Prediction complete.");
  } catch (error) { status("#ann-status", `Error: ${error.message}`); }
  finally {
    button.disabled = !annSelected;
    annSelect.disabled = false;
    document.querySelector("#ann-example").disabled = !annSelected;
    annForm.querySelectorAll("input").forEach(input => { input.disabled = false; });
  }
});

async function loadAnnModels() {
  const response = await fetch("/ann/models");
  if (!response.ok) throw new Error("Could not load ANN models.");
  const catalog = await response.json();
  annModels = catalog.models;
  annSelect.replaceChildren();
  annModels.forEach(model => annSelect.add(new Option(model.display_name, model.model_id)));
  Object.keys(catalog.errors).forEach(id => {
    const option = new Option(`${id} (unavailable)`, id);
    option.disabled = true;
    annSelect.add(option);
  });
  const errors = [catalog.configuration_error, ...Object.entries(catalog.errors).map(([id, message]) => `${id}: ${message}`)].filter(Boolean);
  const errorArea = document.querySelector("#ann-config-errors");
  errorArea.textContent = errors.join(" ");
  errorArea.hidden = errors.length === 0;
  annSelect.value = catalog.default_model || "";
  annSelect.disabled = annSelect.options.length === 0;
  selectAnnModel();
}
loadAnnModels().catch(error => { status("#ann-status", `Error: ${error.message}`); status("#ann-model-summary", "ANN models could not be loaded. Reload this page to try again."); });
