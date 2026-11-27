/* ==========================================================================
   FitScore AI Frontend Interactions & Chart.js Visualizations
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
  if (typeof lucide !== "undefined") {
    lucide.createIcons();
  }

  // Mobile navigation toggle
  const navToggle = document.getElementById("navToggle");
  const navLinks = document.getElementById("navLinks");
  if (navToggle && navLinks) {
    navToggle.addEventListener("click", () => {
      const isOpen = navLinks.classList.toggle("open");
      navToggle.setAttribute("aria-expanded", String(isOpen));
    });
  }

  initDynamicBars();
  initScreeningPage();
  initResultsPage();
});

function initDynamicBars() {
  document.querySelectorAll("[data-width]").forEach((el) => {
    const val = parseFloat(el.getAttribute("data-width"));
    if (!Number.isNaN(val)) {
      el.style.width = `${Math.max(0, Math.min(100, val))}%`;
    }
  });

  document.querySelectorAll("[data-score-pct]").forEach((el) => {
    const val = parseFloat(el.getAttribute("data-score-pct"));
    if (!Number.isNaN(val)) {
      el.style.setProperty("--score-pct", String(Math.max(0, Math.min(100, val))));
    }
  });
}

function initScreeningPage() {
  const form = document.getElementById("screeningForm");
  if (!form) return;

  const jdTextarea = document.getElementById("jobDescription");
  const jdWordCount = document.getElementById("jdWordCount");
  const jdHint = document.getElementById("jdValidationHint");
  const clearJdBtn = document.getElementById("clearJdBtn");
  const fillSampleJdBtn = document.getElementById("fillSampleJdBtn");

  const dropzone = document.getElementById("dropzone");
  const resumeInput = document.getElementById("resumeInput");
  const fileList = document.getElementById("fileList");
  const fileCountPill = document.getElementById("fileCountPill");
  const clearFilesBtn = document.getElementById("clearFilesBtn");
  const clientErrorBox = document.getElementById("clientErrorBox");
  const clientErrorText = document.getElementById("clientErrorText");
  const analyzeBtn = document.getElementById("analyzeBtn");
  const loadingOverlay = document.getElementById("loadingOverlay");

  let selectedFiles = [];
  let isSubmitting = false;

  const sampleJD =
    "We are hiring a Senior Machine Learning Engineer. " +
    "Required skills: Python, Machine Learning, AWS, Docker, SQL, Scikit-learn. " +
    "Preferred: Kubernetes, PyTorch, NLP. " +
    "Minimum 3+ years of experience and at least a Bachelor's degree.";

  function showClientError(message) {
    clientErrorText.textContent = message;
    clientErrorBox.classList.remove("hidden");
  }

  function hideClientError() {
    clientErrorBox.classList.add("hidden");
    clientErrorText.textContent = "";
  }

  function updateJdCount() {
    const text = jdTextarea.value.trim();
    const words = text ? text.split(/\s+/).length : 0;
    jdWordCount.textContent = `${words} word${words === 1 ? "" : "s"}`;
    if (text.length >= 20 && words >= 5) {
      jdHint.textContent = "Job description looks ready for analysis.";
      jdHint.style.color = "#16a34a";
    } else {
      jdHint.textContent = "Minimum 20 characters and 5 words required.";
      jdHint.style.color = "#64748b";
    }
  }

  function syncInputFiles() {
    const dt = new DataTransfer();
    selectedFiles.forEach((f) => dt.items.add(f));
    resumeInput.files = dt.files;
  }

  function formatBytes(bytes) {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  }

  function renderFileList() {
    const count = selectedFiles.length;
    fileCountPill.textContent = `${count} file${count === 1 ? "" : "s"} selected`;
    fileList.innerHTML = "";

    if (count === 0) {
      clearFilesBtn.classList.add("hidden");
      fileList.innerHTML = `
        <li class="file-empty" id="fileEmptyState">
          <i data-lucide="files"></i>
          <span>No resumes uploaded yet.</span>
        </li>
      `;
      if (typeof lucide !== "undefined") lucide.createIcons();
      return;
    }

    clearFilesBtn.classList.remove("hidden");
    selectedFiles.forEach((file, index) => {
      const li = document.createElement("li");
      li.className = "file-item";
      li.innerHTML = `
        <div class="file-item-info">
          <i data-lucide="file-text"></i>
          <span class="file-item-name" title="${escapeHtml(file.name)}">${escapeHtml(file.name)}</span>
          <span class="file-item-size">(${formatBytes(file.size)})</span>
        </div>
        <button type="button" class="file-remove-btn" data-index="${index}" aria-label="Remove ${escapeHtml(file.name)}">
          <i data-lucide="x"></i>
        </button>
      `;
      fileList.appendChild(li);
    });

    fileList.querySelectorAll(".file-remove-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const idx = Number(btn.getAttribute("data-index"));
        selectedFiles.splice(idx, 1);
        syncInputFiles();
        renderFileList();
        hideClientError();
      });
    });

    if (typeof lucide !== "undefined") lucide.createIcons();
  }

  function addFiles(fileCollection) {
    hideClientError();
    const rejectedNames = [];

    Array.from(fileCollection).forEach((file) => {
      const lower = file.name.toLowerCase();
      if (!lower.endsWith(".pdf") && !lower.endsWith(".docx")) {
        rejectedNames.push(file.name);
        return;
      }
      const alreadyAdded = selectedFiles.some(
        (existing) => existing.name === file.name && existing.size === file.size
      );
      if (!alreadyAdded) {
        selectedFiles.push(file);
      }
    });

    if (rejectedNames.length > 0) {
      showClientError(
        `Unsupported file format skipped: ${rejectedNames.join(", ")}. Only PDF (.pdf) and Word (.docx) files are accepted.`
      );
    }

    syncInputFiles();
    renderFileList();
  }

  jdTextarea.addEventListener("input", updateJdCount);
  updateJdCount();

  clearJdBtn.addEventListener("click", () => {
    jdTextarea.value = "";
    updateJdCount();
    hideClientError();
    jdTextarea.focus();
  });

  fillSampleJdBtn.addEventListener("click", () => {
    jdTextarea.value = sampleJD;
    updateJdCount();
    hideClientError();
  });

  dropzone.addEventListener("click", () => resumeInput.click());
  dropzone.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();
      resumeInput.click();
    }
  });

  resumeInput.addEventListener("change", () => {
    if (resumeInput.files.length > 0) {
      addFiles(resumeInput.files);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer && e.dataTransfer.files.length > 0) {
      addFiles(e.dataTransfer.files);
    }
  });

  clearFilesBtn.addEventListener("click", () => {
    selectedFiles = [];
    syncInputFiles();
    renderFileList();
    hideClientError();
  });

  form.addEventListener("submit", (e) => {
    if (isSubmitting) {
      e.preventDefault();
      return;
    }

    const jdText = jdTextarea.value.trim();
    const jdWords = jdText ? jdText.split(/\s+/).length : 0;

    if (!jdText || jdText.length < 20 || jdWords < 5) {
      e.preventDefault();
      showClientError("Please enter a complete job description (at least 20 characters and 5 words).");
      jdTextarea.focus();
      return;
    }

    if (selectedFiles.length === 0) {
      e.preventDefault();
      showClientError("Please upload at least one PDF or DOCX resume before analyzing.");
      return;
    }

    isSubmitting = true;
    analyzeBtn.disabled = true;
    loadingOverlay.classList.remove("hidden");
  });
}

function initResultsPage() {
  const searchInput = document.getElementById("candidateSearch");
  const categoryFilter = document.getElementById("categoryFilter");
  const table = document.getElementById("rankingTable");
  const noMatchBox = document.getElementById("noFilterMatch");

  if (table && searchInput && categoryFilter) {
    const rows = Array.from(table.querySelectorAll("tbody tr"));

    function applyTableFilters() {
      const query = searchInput.value.trim().toLowerCase();
      const selectedCat = categoryFilter.value;
      let visibleCount = 0;

      rows.forEach((row) => {
        const name = row.getAttribute("data-name") || "";
        const cat = row.getAttribute("data-category") || "";
        const matchesName = name.includes(query);
        const matchesCat = selectedCat === "all" || cat === selectedCat;

        if (matchesName && matchesCat) {
          row.classList.remove("hidden");
          visibleCount += 1;
        } else {
          row.classList.add("hidden");
        }
      });

      if (visibleCount === 0) {
        noMatchBox.classList.remove("hidden");
      } else {
        noMatchBox.classList.add("hidden");
      }
    }

    searchInput.addEventListener("input", applyTableFilters);
    categoryFilter.addEventListener("change", applyTableFilters);
  }

  const analyticsTag = document.getElementById("analyticsData");
  if (!analyticsTag || typeof Chart === "undefined") return;

  const rawJson = analyticsTag.getAttribute("data-json");
  if (!rawJson) return;
  const data = JSON.parse(rawJson);

  Chart.defaults.font.family = '"Inter", -apple-system, BlinkMacSystemFont, sans-serif';
  Chart.defaults.color = "#475569";

  // 1. Candidate Score Comparison (Horizontal Bar)
  const scoreCanvas = document.getElementById("candidateScoreChart");
  if (scoreCanvas) {
    new Chart(scoreCanvas, {
      type: "bar",
      data: {
        labels: data.candidate_names,
        datasets: [{
          label: "Predicted Score (0–100)",
          data: data.candidate_scores,
          backgroundColor: "#2563eb",
          borderRadius: 6,
        }],
      },
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          x: { min: 0, max: 100, grid: { color: "#f1f5f9" } },
          y: { grid: { display: false } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  // 2. Suitability Category Distribution (Doughnut)
  const catCanvas = document.getElementById("categoryDistChart");
  if (catCanvas) {
    new Chart(catCanvas, {
      type: "doughnut",
      data: {
        labels: data.category_labels,
        datasets: [{
          data: data.category_counts,
          backgroundColor: ["#dc2626", "#f97316", "#eab308", "#2563eb", "#16a34a"],
          borderWidth: 2,
          borderColor: "#ffffff",
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: "bottom" },
        },
      },
    });
  }

  // 3. Suitability Score Distribution (Bins 0-20 .. 81-100)
  const distCanvas = document.getElementById("scoreDistChart");
  if (distCanvas) {
    new Chart(distCanvas, {
      type: "bar",
      data: {
        labels: data.score_bin_labels,
        datasets: [{
          label: "Candidates",
          data: data.score_bin_counts,
          backgroundColor: "#6366f1",
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { beginAtZero: true, ticks: { stepSize: 1 }, grid: { color: "#f1f5f9" } },
          x: { grid: { display: false } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }

  // 4. Required Skill Match Comparison (%)
  const skillCanvas = document.getElementById("skillMatchChart");
  if (skillCanvas) {
    new Chart(skillCanvas, {
      type: "bar",
      data: {
        labels: data.candidate_names,
        datasets: [{
          label: "Required Skill Match (%)",
          data: data.candidate_skill_matches,
          backgroundColor: "#1e3a8a",
          borderRadius: 6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
          y: { min: 0, max: 100, grid: { color: "#f1f5f9" } },
          x: { grid: { display: false } },
        },
        plugins: { legend: { display: false } },
      },
    });
  }
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}
