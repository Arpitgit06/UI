// OmniUI Frontend Logic
document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const systemStatusPill = document.getElementById("systemStatusPill");
  const systemStatusText = document.getElementById("systemStatusText");
  const gpuNameDisplay = document.getElementById("gpuNameDisplay");
  const vramUsageDisplay = document.getElementById("vramUsageDisplay");
  const queueSizeDisplay = document.getElementById("queueSizeDisplay");
  const activeJobDisplay = document.getElementById("activeJobDisplay");
  const llmModelDisplay = document.getElementById("llmModelDisplay");
  const llmTargetDisplay = document.getElementById("llmTargetDisplay");

  const dropzone = document.getElementById("dropzone");
  const videoFileInput = document.getElementById("videoFileInput");
  const dropzoneDefault = document.getElementById("dropzoneDefault");
  const dropzonePreview = document.getElementById("dropzonePreview");
  const previewFileName = document.getElementById("previewFileName");
  const previewFileMeta = document.getElementById("previewFileMeta");
  const clearFileBtn = document.getElementById("clearFileBtn");
  const startPipelineBtn = document.getElementById("startPipelineBtn");

  const pipelineSection = document.getElementById("pipelineSection");
  const jobIdBadge = document.getElementById("jobIdBadge");
  const jobStatusBadge = document.getElementById("jobStatusBadge");
  const keyStatesStat = document.getElementById("keyStatesStat");
  const stageDetailText = document.getElementById("stageDetailText");
  const lastUpdatedTime = document.getElementById("lastUpdatedTime");

  const completionActions = document.getElementById("completionActions");
  const downloadZipBtn = document.getElementById("downloadZipBtn");
  const resetAppBtn = document.getElementById("resetAppBtn");
  const errorBox = document.getElementById("errorBox");
  const errorMessageText = document.getElementById("errorMessageText");

  const steps = {
    parsing_video: document.getElementById("stepA"),
    detecting_elements: document.getElementById("stepB"),
    synthesizing_dom: document.getElementById("stepC"),
    generating_code: document.getElementById("stepD"),
    packaging: document.getElementById("stepE")
  };

  const stepOrder = [
    "parsing_video",
    "detecting_elements",
    "synthesizing_dom",
    "generating_code",
    "packaging"
  ];

  let selectedFile = null;
  let activeJobId = null;
  let pollInterval = null;

  // 1. Health Poll & System Monitor
  async function fetchHealth() {
    try {
      const res = await fetch("/health");
      if (!res.ok) throw new Error("Health check failed");
      const data = await res.json();

      systemStatusText.textContent = "Service Online & Ready";
      const dot = systemStatusPill.querySelector(".status-dot");
      if (dot) dot.style.backgroundColor = "var(--accent-green)";

      queueSizeDisplay.textContent = `${data.queue_size || 0} Pending`;
      activeJobDisplay.textContent = `Active: ${data.active_job || "None"}`;

      if (data.gpu && data.gpu.cuda_available) {
        gpuNameDisplay.textContent = data.gpu.device_name || "NVIDIA CUDA GPU";
        vramUsageDisplay.textContent = data.gpu.memory || "VRAM available";
      } else {
        gpuNameDisplay.textContent = "CPU Fallback Mode";
        vramUsageDisplay.textContent = "No CUDA device detected";
      }

      if (data.local_llm) {
        llmModelDisplay.textContent = "Local LLM (Active)";
        llmTargetDisplay.textContent = `Model: ${data.local_llm.model || "Unknown"} (${data.local_llm.quantization || "full"})`;
      }
    } catch (err) {
      console.error("Health poll error:", err);
      systemStatusText.textContent = "Server Offline / Reconnecting...";
      const dot = systemStatusPill.querySelector(".status-dot");
      if (dot) dot.style.backgroundColor = "var(--accent-red)";
    }
  }

  fetchHealth();
  setInterval(fetchHealth, 5000);

  // 2. Drag and Drop File Handlers
  dropzone.addEventListener("click", (e) => {
    if (e.target !== clearFileBtn && !clearFileBtn.contains(e.target)) {
      videoFileInput.click();
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  });

  videoFileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileSelection(e.target.files[0]);
    }
  });

  clearFileBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    resetFileSelection();
  });

  function handleFileSelection(file) {
    const validTypes = ["video/mp4", "video/webm", "video/x-matroska"];
    if (!validTypes.includes(file.type) && !file.name.endsWith(".mp4") && !file.name.endsWith(".webm") && !file.name.endsWith(".mkv")) {
      alert("Unsupported file format. Please upload an MP4, WebM, or MKV video recording.");
      return;
    }

    selectedFile = file;
    previewFileName.textContent = file.name;
    const sizeMb = (file.size / (1024 * 1024)).toFixed(2);
    previewFileMeta.textContent = `Ready to convert • ${sizeMb} MB`;

    dropzoneDefault.classList.add("hidden");
    dropzonePreview.classList.remove("hidden");
    startPipelineBtn.disabled = false;
  }

  function resetFileSelection() {
    selectedFile = null;
    videoFileInput.value = "";
    dropzonePreview.classList.add("hidden");
    dropzoneDefault.classList.remove("hidden");
    startPipelineBtn.disabled = true;
  }

  // 3. Start Conversion & Job Submission
  startPipelineBtn.addEventListener("click", async () => {
    if (!selectedFile) return;

    startPipelineBtn.disabled = true;
    startPipelineBtn.innerHTML = `<span>Uploading Video...</span>`;

    const formData = new FormData();
    formData.append("video", selectedFile);

    try {
      const res = await fetch("/jobs", {
        method: "POST",
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        throw new Error(errData.detail || `Upload failed with status ${res.status}`);
      }

      const job = await res.json();
      activeJobId = job.job_id;
      
      // Show Tracker
      jobIdBadge.textContent = `ID: #${job.job_id}`;
      pipelineSection.classList.remove("hidden");
      pipelineSection.scrollIntoView({ behavior: "smooth" });

      startPollingJob(job.job_id);
    } catch (err) {
      alert(`Error submitting job: ${err.message}`);
      startPipelineBtn.disabled = false;
      startPipelineBtn.innerHTML = `<span>Start Local Conversion</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>`;
    }
  });

  // 4. Job Polling & Stepper Updates
  function startPollingJob(jobId) {
    if (pollInterval) clearInterval(pollInterval);
    
    pollInterval = setInterval(async () => {
      try {
        const res = await fetch(`/jobs/${jobId}`);
        if (!res.ok) throw new Error("Could not fetch job status");
        const job = await res.json();
        updateUIForJob(job);

        if (job.status === "complete" || job.status === "failed") {
          clearInterval(pollInterval);
          pollInterval = null;
          fetchHealth();
        }
      } catch (err) {
        console.error("Polling error:", err);
      }
    }, 1500);

    // Initial check right away
    fetch(`/jobs/${jobId}`).then(r => r.json()).then(updateUIForJob).catch(console.error);
  }

  function updateUIForJob(job) {
    // Status badge
    jobStatusBadge.textContent = job.status.toUpperCase();
    jobStatusBadge.className = `status-badge ${job.status === 'queued' ? 'queued' : (job.status === 'complete' ? 'complete' : (job.status === 'failed' ? 'failed' : 'running'))}`;

    // Stats and detail log
    keyStatesStat.textContent = job.key_states_detected || 0;
    stageDetailText.textContent = job.current_stage_detail || `Current stage: ${job.status}`;
    lastUpdatedTime.textContent = new Date().toLocaleTimeString();

    // Update 5-stage stepper
    let currentIdx = stepOrder.indexOf(job.status);
    if (job.status === "complete") currentIdx = stepOrder.length;

    stepOrder.forEach((stageName, idx) => {
      const card = steps[stageName];
      if (!card) return;

      const statusLbl = card.querySelector(".step-status");
      if (job.status === "complete" || idx < currentIdx) {
        card.className = "step-card completed";
        if (statusLbl) statusLbl.textContent = "Completed ✓";
      } else if (idx === currentIdx && job.status !== "failed") {
        card.className = "step-card active";
        if (statusLbl) statusLbl.textContent = "Processing... • Active VRAM Scope";
      } else {
        card.className = "step-card";
        if (statusLbl) statusLbl.textContent = "Waiting";
      }
    });

    if (job.status === "complete") {
      completionActions.classList.remove("hidden");
      downloadZipBtn.href = `/jobs/${job.job_id}/download`;
      errorBox.classList.add("hidden");
    } else if (job.status === "failed") {
      completionActions.classList.add("hidden");
      errorBox.classList.remove("hidden");
      errorMessageText.textContent = job.error_message || "An unknown error occurred during pipeline processing.";
    } else {
      completionActions.classList.add("hidden");
      errorBox.classList.add("hidden");
    }
  }

  // 5. Reset button
  resetAppBtn.addEventListener("click", () => {
    resetFileSelection();
    pipelineSection.classList.add("hidden");
    completionActions.classList.add("hidden");
    errorBox.classList.add("hidden");
    if (pollInterval) clearInterval(pollInterval);
    activeJobId = null;
    startPipelineBtn.disabled = false;
    startPipelineBtn.innerHTML = `<span>Start Local Conversion</span><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 5l7 7m0 0l-7 7m7-7H3"/></svg>`;
    window.scrollTo({ top: 0, behavior: "smooth" });
  });
});
