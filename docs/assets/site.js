(async function main() {
  const page = document.body.dataset.page || "home";
  const data = await fetchJson("site-data/project_site.json");
  if (!data) return;

  const abstractEl = document.getElementById("project-abstract");
  if (abstractEl) abstractEl.textContent = data.project.abstract;

  const nameEl = document.getElementById("project-name");
  if (nameEl) nameEl.textContent = data.project.name;

  if (page === "home") {
    renderHeroLinks(data.docs_links, data.project.repo_url);
    renderMetricCards("hero-metrics", data.hero_metrics);
    renderBulletCards("contributions", data.contributions);
    renderComparisons("comparison-rows", data.comparisons);
    renderBenchmarks("benchmark-grid", data);
    renderAudienceCards("audience-grid", data.audiences);
    renderFigureCards("figure-grid", data.figures);
    renderDocCards("docs-links", data.docs_links);
  }

  if (page === "about") {
    renderBulletCards("contributions", data.contributions);
    renderDocList("docs-links-list", data.docs_links);
  }

  if (page === "documents") {
    renderMetricCards("markdown-metrics", [
      { label: "Tracked markdown files", value: String(data.markdown_stats.total) },
      { label: "Published on GitHub Pages", value: String(data.markdown_stats.published) },
      { label: "GitHub-linked references", value: String(data.markdown_stats.repo_only) },
    ]);
    renderMarkdownCatalog("markdown-library", data.markdown_catalog);
  }
})();

async function fetchJson(path) {
  try {
    const response = await fetch(path);
    if (!response.ok) throw new Error(`Failed to load ${path}`);
    return await response.json();
  } catch (error) {
    console.error(error);
    return null;
  }
}

function renderHeroLinks(links, repoUrl) {
  const target = document.getElementById("hero-links");
  if (!target) return;
  const primary = [
    { label: "Read the final report (PDF)", href: "final_report_pack/LayerForge_X_Final_Report_2026_04_22.pdf", className: "button-link" },
    { label: "Read the final report (DOCX)", href: "final_report_pack/LayerForge_X_Final_Report_2026_04_22.docx", className: "button-link button-link--secondary" },
    { label: "Inspect the evidence pack", href: "RESULTS_SUMMARY_CURRENT.md", className: "button-link button-link--secondary" },
    { label: "Open local web UI", href: "webui.html", className: "button-link button-link--secondary" },
    { label: "View source on GitHub", href: repoUrl, className: "button-link button-link--secondary" },
  ];
  target.innerHTML = primary
    .map((item) => `<a class="${item.className}" href="${item.href}">${escapeHtml(item.label)}</a>`)
    .join("");
}

function renderMetricCards(id, items) {
  const target = document.getElementById(id);
  if (!target) return;
  target.innerHTML = items
    .map(
      (item) => `
        <article class="metric-card">
          <span class="metric-card__label">${escapeHtml(item.label)}</span>
          <strong class="metric-card__value">${escapeHtml(String(item.value))}</strong>
        </article>
      `,
    )
    .join("");
}

function renderBulletCards(id, items) {
  const target = document.getElementById(id);
  if (!target) return;
  target.innerHTML = items
    .map((item) => {
      const title = typeof item === "string" ? "Contribution" : item.title;
      const body = typeof item === "string" ? item : item.body;
      return `
        <article class="bullet-card">
          <div class="bullet-card__title">${escapeHtml(title)}</div>
          <div class="bullet-card__body">${escapeHtml(body)}</div>
        </article>
      `;
    })
    .join("");
}

function renderAudienceCards(id, items) {
  const target = document.getElementById(id);
  if (!target) return;
  target.innerHTML = items
    .map(
      (item) => `
        <article class="audience-card">
          <div class="audience-card__title">${escapeHtml(item.title)}</div>
          <div class="audience-card__body">${escapeHtml(item.body)}</div>
        </article>
      `,
    )
    .join("");
}

function renderComparisons(id, items) {
  const target = document.getElementById(id);
  if (!target) return;
  target.innerHTML = items
    .map(
      (item) => `
        <tr>
          <td><strong>${escapeHtml(item.label)}</strong></td>
          <td>${escapeHtml(String(item.images))}</td>
          <td>${Number(item.mean_psnr).toFixed(4)}</td>
          <td>${Number(item.mean_ssim).toFixed(4)}</td>
          <td>${Number(item.mean_self_eval_score).toFixed(4)}</td>
          <td>${escapeHtml(item.description)}</td>
        </tr>
      `,
    )
    .join("");
}

function renderBenchmarks(id, data) {
  const target = document.getElementById(id);
  if (!target) return;
  const cards = [
    {
      label: "Prompt extraction",
      value: `${(data.benchmarks.prompt_extract.text_hit_rate * 100).toFixed(0)}% text hit rate`,
      body: `${data.benchmarks.prompt_extract.queries_per_type} queries per prompt type. ${data.benchmarks.prompt_extract.note}`,
    },
    {
      label: "Transparent benchmark",
      value: `${data.benchmarks.transparent.alpha_mae.toFixed(4)} alpha MAE`,
      body: `Background PSNR ${data.benchmarks.transparent.background_psnr.toFixed(2)}, SSIM ${data.benchmarks.transparent.background_ssim.toFixed(4)}. ${data.benchmarks.transparent.note}`,
    },
    {
      label: "Associated-effect prototype",
      value: `${data.benchmarks.effects.effect_iou.toFixed(4)} IoU`,
      body: `${data.benchmarks.effects.predicted_pixels} predicted effect pixels vs ${data.benchmarks.effects.ground_truth_pixels} ground-truth pixels. ${data.benchmarks.effects.note}`,
    },
    {
      label: "Public grouping benchmarks",
      value: `${data.public_benchmarks.ade20k_miou.toFixed(3)} ADE20K mIoU`,
      body: `COCO Panoptic supported-group mIoU ${data.public_benchmarks.coco_panoptic_miou.toFixed(3)}. DIODE scaled Depth Pro abs-rel ${data.public_benchmarks.diode_depthpro_scale_abs_rel.toFixed(3)}.`,
    },
  ];
  target.innerHTML = cards
    .map(
      (item) => `
        <article class="benchmark-card">
          <div class="benchmark-card__label">${escapeHtml(item.label)}</div>
          <strong class="benchmark-card__value">${escapeHtml(item.value)}</strong>
          <div class="benchmark-card__body">${escapeHtml(item.body)}</div>
        </article>
      `,
    )
    .join("");
}

function renderFigureCards(id, items) {
  const target = document.getElementById(id);
  if (!target) return;
  target.innerHTML = items
    .map(
      (item) => `
        <article class="figure-card">
          <a href="${item.path}" target="_blank" rel="noreferrer">
            <img src="${item.path}" alt="${escapeHtml(item.title)}" loading="lazy" />
            <div class="figure-card__title">${escapeHtml(item.title)}</div>
            <div class="figure-card__body">${escapeHtml(item.caption)}</div>
          </a>
        </article>
      `,
    )
    .join("");
}

function renderDocCards(id, items) {
  const target = document.getElementById(id);
  if (!target) return;
  target.innerHTML = items
    .map(
      (item) => `
        <article class="doc-card">
          <a href="${item.href}">
            <div class="doc-card__title">${escapeHtml(item.label)}</div>
            <div class="doc-card__body">${escapeHtml(item.href)}</div>
          </a>
        </article>
      `,
    )
    .join("");
}

function renderDocList(id, items) {
  const target = document.getElementById(id);
  if (!target) return;
  target.innerHTML = items
    .map((item) => `<li><a href="${item.href}">${escapeHtml(item.label)}</a></li>`)
    .join("");
}

function renderMarkdownCatalog(id, items) {
  const target = document.getElementById(id);
  if (!target) return;

  const grouped = new Map();
  for (const item of items) {
    const groupItems = grouped.get(item.group) || [];
    groupItems.push(item);
    grouped.set(item.group, groupItems);
  }

  target.innerHTML = Array.from(grouped.entries())
    .map(([group, groupItems]) => {
      const eyebrow = groupItems[0]?.published ? "Published pages" : "Repository references";
      const cards = groupItems
        .map((item) => {
          const externalAttrs = item.published ? "" : ' target="_blank" rel="noreferrer"';
          return `
            <article class="doc-card">
              <a href="${item.href}"${externalAttrs}>
                <div class="doc-card__title">${escapeHtml(item.label)}</div>
                <div class="doc-card__body">${escapeHtml(item.source_path)}</div>
                <div class="doc-card__body">${escapeHtml(item.surface_label)}</div>
              </a>
            </article>
          `;
        })
        .join("");

      return `
        <section class="section-block prose-block">
          <div class="section-head">
            <p class="eyebrow">${escapeHtml(eyebrow)}</p>
            <h2>${escapeHtml(group)}</h2>
          </div>
          <div class="doc-link-grid">
            ${cards}
          </div>
        </section>
      `;
    })
    .join("");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
