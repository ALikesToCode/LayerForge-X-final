(async function boot() {
  const runtime = window.__LAYERFORGE_RUNTIME__ === true ? await fetchJson("/api/runtime") : null;
  const project = runtime && runtime.available
    ? await fetchJson("/api/project-summary")
    : await fetchJson("site-data/project_site.json");
  if (project) renderRuntimeStatus(runtime, project);
  wireForm(runtime);
})();

async function fetchJson(path) {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}`);
    return await response.json();
  } catch (error) {
    console.warn(error);
    return null;
  }
}

function renderRuntimeStatus(runtime, project) {
  const target = document.getElementById("runtime-status");
  const runButton = document.getElementById("run-button");
  const runHint = document.getElementById("run-hint");
  const entrypoint = project && project.local_lab && project.local_lab.entrypoint
    ? project.local_lab.entrypoint
    : "layerforge webui --open-browser";
  if (!target) return;
  if (runtime && runtime.available) {
    target.innerHTML = `
      <article class="status-card">
        <span class="metric-card__label">Runtime</span>
        <strong class="metric-card__value">Local server ready</strong>
        <div class="benchmark-card__body">Root: <code>${escapeHtml(runtime.repo_root)}</code></div>
      </article>
      <article class="status-card">
        <span class="metric-card__label">Recommended command</span>
        <strong class="metric-card__value"><code>${escapeHtml(entrypoint)}</code></strong>
      </article>
    `;
    if (runButton) runButton.disabled = false;
    if (runHint) {
      runHint.textContent = "Local runtime is active. Upload an image and select a mode.";
    }
    return;
  }
  target.innerHTML = `
    <article class="status-card">
      <span class="metric-card__label">Runtime</span>
      <strong class="metric-card__value">Static Pages mode</strong>
      <div class="benchmark-card__body">The public site can browse the evidence pack, but local runs require the Python web UI server.</div>
    </article>
    <article class="status-card">
      <span class="metric-card__label">Local launch</span>
      <strong class="metric-card__value"><code>${escapeHtml(entrypoint)}</code></strong>
      <div class="benchmark-card__body">Use the local runtime to upload images, dispatch runs, and inspect generated artifacts from the browser.</div>
    </article>
  `;
  if (runHint) {
    runHint.textContent = `Static overview only. Start the local server with ${entrypoint}.`;
  }
}

function wireForm(runtime) {
  const form = document.getElementById("webui-form");
  if (!form) return;
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!runtime || !runtime.available) return;

    const status = document.getElementById("run-status");
    const button = document.getElementById("run-button");
    const fileInput = form.elements.image;
    const file = fileInput.files && fileInput.files[0];
    if (!file) {
      setStatus("Select an input image before starting a local run.", "error");
      return;
    }

    button.disabled = true;
    setStatus("Running the LayerForge pipeline locally. The browser request will stay open until the selected mode completes.", "pending");

    try {
      const imageBase64 = await fileToBase64(file);
        const payload = {
            mode: form.elements.mode.value,
            filename: file.name,
            image_base64: imageBase64,
        config: form.elements.config.value,
        segmenter: emptyToNull(form.elements.segmenter.value),
        depth: emptyToNull(form.elements.depth.value),
        prompt: emptyToNull(form.elements.prompt.value),
        point: emptyToNull(form.elements.point.value),
        box: emptyToNull(form.elements.box.value),
        prompt_source: emptyToNull(form.elements.prompt_source.value),
        ordering: emptyToNull(form.elements.ordering.value),
            device: emptyToNull(form.elements.device.value) || "auto",
            max_layers: emptyToNull(form.elements.max_layers.value),
            no_parallax: form.elements.no_parallax.checked,
            use_frontier_base: form.elements.use_frontier_base.checked,
        };
      const response = await fetch("/api/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || "LayerForge run failed");
      renderResult(result);
      setStatus(`Completed ${result.mode} for ${result.input_filename}.`, "success");
    } catch (error) {
      setStatus(error.message || String(error), "error");
    } finally {
      button.disabled = false;
    }
  });
}

function renderResult(result) {
  const metricsTarget = document.getElementById("result-metrics");
  const linksTarget = document.getElementById("result-links");
  const previewsTarget = document.getElementById("result-previews");

  const metricEntries = Object.entries(result.summary_metrics || {}).map(([key, value]) => ({
    label: key.replaceAll("_", " "),
    value: typeof value === "number" ? value.toFixed(4) : JSON.stringify(value),
  }));
  metricsTarget.innerHTML = metricEntries
    .map(
      (item) => `
        <article class="benchmark-card">
          <div class="benchmark-card__label">${escapeHtml(item.label)}</div>
          <strong class="benchmark-card__value">${escapeHtml(item.value)}</strong>
        </article>
      `,
    )
    .join("");

  linksTarget.innerHTML = Object.entries(result.urls || {})
    .filter(([, value]) => value)
    .map(
      ([key, value]) => `
        <article class="doc-card result-link">
          <a href="${value}" target="_blank" rel="noreferrer">
            <div class="doc-card__title">${escapeHtml(key)}</div>
            <div class="doc-card__body">${escapeHtml(value)}</div>
          </a>
        </article>
      `,
    )
    .join("");

  previewsTarget.innerHTML = (result.previews || [])
    .map(
      (item) => `
        <article class="figure-card">
          <a href="${item.url}" target="_blank" rel="noreferrer">
            <img src="${item.url}" alt="${escapeHtml(item.label)}" loading="lazy" />
            <div class="figure-card__title">${escapeHtml(item.label)}</div>
          </a>
        </article>
      `,
    )
    .join("");

  renderInspector(result.inspector || null);
}

function renderInspector(inspector) {
  const target = document.getElementById("result-inspector");
  if (!target) return;
  if (!inspector || !(inspector.layers || []).length) {
    target.innerHTML = "";
    return;
  }

  const validation = inspector.diagnostics && inspector.diagnostics.validation
    ? inspector.diagnostics.validation
    : {};
  const heatmap = inspector.diagnostics ? inspector.diagnostics.error_heatmap_url : null;
  const graphUrl = inspector.diagnostics ? inspector.diagnostics.graph_url : null;
  const validationUrl = inspector.diagnostics ? inspector.diagnostics.validation_url : null;
  target.innerHTML = `
    <section class="inspector-panel">
      <div class="inspector-panel__head">
        <div>
          <p class="eyebrow">Layer inspector</p>
          <h3>Depth, masks, evidence, and validation</h3>
        </div>
        <div class="inspector-diagnostics">
          <span class="inspector-pill ${validation.ok === false ? "is-error" : "is-ok"}">
            validation ${validation.ok === false ? "failed" : "ok"}
          </span>
          ${graphUrl ? `<a href="${graphUrl}" target="_blank" rel="noreferrer">graph JSON</a>` : ""}
          ${validationUrl ? `<a href="${validationUrl}" target="_blank" rel="noreferrer">validation JSON</a>` : ""}
        </div>
      </div>
      ${heatmap ? `
        <a class="inspector-heatmap" href="${heatmap}" target="_blank" rel="noreferrer">
          <img src="${heatmap}" alt="Recomposition error heatmap" loading="lazy" />
          <span>Recomposition error heatmap</span>
        </a>
      ` : ""}
      <div class="inspector-layer-grid">
        ${(inspector.layers || []).map(renderInspectorLayer).join("")}
      </div>
      ${renderEdgeEvidence(inspector.edges || [])}
    </section>
  `;
  for (const toggle of target.querySelectorAll("[data-layer-toggle]")) {
    toggle.addEventListener("change", () => {
      const card = toggle.closest(".inspector-layer-card");
      if (card) card.classList.toggle("is-muted", !toggle.checked);
    });
  }
}

function renderInspectorLayer(layer) {
  const assets = layer.assets || {};
  const depthStats = Object.entries(layer.depth_stats || {})
    .map(([key, value]) => `<span><b>${escapeHtml(key.replaceAll("_", " "))}</b>${formatInspectorValue(value)}</span>`)
    .join("");
  const quality = Object.entries(layer.quality_metrics || {})
    .filter(([, value]) => value !== null && value !== undefined && typeof value !== "object")
    .slice(0, 6)
    .map(([key, value]) => `<span><b>${escapeHtml(key.replaceAll("_", " "))}</b>${formatInspectorValue(value)}</span>`)
    .join("");
  const assetLinks = Object.entries(assets)
    .filter(([, url]) => url)
    .map(([key, url]) => `<a href="${url}" target="_blank" rel="noreferrer">${escapeHtml(key)}</a>`)
    .join("");
  return `
    <article class="inspector-layer-card">
      <label class="inspector-toggle">
        <input type="checkbox" checked data-layer-toggle />
        <span>${escapeHtml(layer.name || "layer")}</span>
      </label>
      <div class="inspector-layer-card__meta">
        <span>${escapeHtml(layer.group || "unknown")}</span>
        <span>rank ${escapeHtml(layer.rank ?? "")}</span>
        <span>${escapeHtml(layer.label || "")}</span>
      </div>
      <div class="inspector-stat-grid">${depthStats}</div>
      <div class="inspector-stat-grid">${quality}</div>
      <div class="inspector-asset-row">${assetLinks}</div>
    </article>
  `;
}

function renderEdgeEvidence(edges) {
  if (!edges.length) return "";
  return `
    <div class="inspector-edge-table">
      <div class="inspector-edge-table__head">
        <span>near</span>
        <span>far</span>
        <span>relation</span>
        <span>confidence</span>
        <span>evidence</span>
      </div>
      ${edges.slice(0, 80).map((edge) => {
        const evidence = Object.entries(edge.evidence || {})
          .filter(([, value]) => value !== null && value !== undefined)
          .map(([key, value]) => `${key.replaceAll("_", " ")}=${formatInspectorValue(value)}`)
          .join("; ");
        return `
          <div class="inspector-edge-table__row">
            <span>${escapeHtml(edge.near_id ?? "")}</span>
            <span>${escapeHtml(edge.far_id ?? "")}</span>
            <span>${escapeHtml(edge.relation || "uncertain")}</span>
            <span>${formatInspectorValue(edge.confidence)}</span>
            <span>${escapeHtml(evidence)}</span>
          </div>
        `;
      }).join("")}
    </div>
  `;
}

function formatInspectorValue(value) {
  if (typeof value === "number") return Number.isFinite(value) ? value.toFixed(4) : "";
  if (value === null || value === undefined) return "";
  return escapeHtml(String(value));
}

function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = String(reader.result || "");
      const comma = result.indexOf(",");
      resolve(comma >= 0 ? result.slice(comma + 1) : result);
    };
    reader.onerror = () => reject(reader.error || new Error("Failed to read upload"));
    reader.readAsDataURL(file);
  });
}

function emptyToNull(value) {
  const text = String(value || "").trim();
  return text ? text : null;
}

function setStatus(message, state) {
  const status = document.getElementById("run-status");
  status.className = "status-banner";
  if (state === "error") status.classList.add("is-error");
  if (state === "pending") status.classList.add("is-pending");
  status.textContent = message;
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
