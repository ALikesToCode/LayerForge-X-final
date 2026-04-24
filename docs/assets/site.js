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

  if (page === "reader") {
    await renderMarkdownReader(data);
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
    { label: "Inspect the evidence pack", href: "reader.html?path=RESULTS_SUMMARY_CURRENT.md", className: "button-link button-link--secondary" },
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

async function renderMarkdownReader(data) {
  const titleEl = document.getElementById("reader-title");
  const metaEl = document.getElementById("reader-meta");
  const bodyEl = document.getElementById("reader-body");
  const sourceEl = document.getElementById("reader-source-link");
  const params = new URLSearchParams(window.location.search);
  const requestedPath = normalizeDocPath(params.get("path") || "");

  if (!requestedPath || !requestedPath.toLowerCase().endsWith(".md")) {
    renderReaderError(titleEl, metaEl, bodyEl, "Document not found", "No markdown document was requested.");
    return;
  }

  const item = (data.markdown_catalog || []).find((entry) => entry.docs_path === requestedPath);
  if (!item || !item.source_asset) {
    renderReaderError(
      titleEl,
      metaEl,
      bodyEl,
      "Document not published",
      `${requestedPath} is not in the published markdown catalog.`,
    );
    return;
  }

  try {
    const response = await fetch(encodePathForUrl(item.source_asset));
    if (!response.ok) throw new Error(`Failed to load ${item.source_asset}`);
    const markdown = await response.text();
    document.title = `${item.label} | LayerForge-X`;
    if (titleEl) titleEl.textContent = item.label;
    if (metaEl) metaEl.textContent = `docs/${requestedPath}`;
    if (sourceEl) {
      sourceEl.href = `${data.project.repo_url}/blob/main/docs/${requestedPath}`;
      sourceEl.hidden = false;
    }
    if (bodyEl) bodyEl.innerHTML = renderMarkdown(markdown, requestedPath, data.project.repo_url);
  } catch (error) {
    console.error(error);
    renderReaderError(
      titleEl,
      metaEl,
      bodyEl,
      "Document failed to load",
      `${requestedPath} could not be loaded from the published markdown source bundle.`,
    );
  }
}

function renderReaderError(titleEl, metaEl, bodyEl, title, message) {
  if (titleEl) titleEl.textContent = title;
  if (metaEl) metaEl.textContent = "";
  if (bodyEl) {
    bodyEl.innerHTML = `<p>${escapeHtml(message)}</p><p><a href="documents.html">Open the markdown library</a></p>`;
  }
}

function renderMarkdown(markdown, currentPath, repoUrl) {
  const lines = markdown.replace(/\r\n?/g, "\n").split("\n");
  const html = [];
  let index = 0;

  while (index < lines.length) {
    const line = lines[index];

    if (!line.trim()) {
      index += 1;
      continue;
    }

    const fence = line.match(/^\s*```(\S+)?\s*$/);
    if (fence) {
      const language = fence[1] || "";
      index += 1;
      const code = [];
      while (index < lines.length && !/^\s*```\s*$/.test(lines[index])) {
        code.push(lines[index]);
        index += 1;
      }
      if (index < lines.length) index += 1;
      html.push(
        `<pre><code class="${language ? `language-${escapeHtml(language)}` : ""}">${escapeHtml(code.join("\n"))}</code></pre>`,
      );
      continue;
    }

    const htmlHeading = line.match(/^\s*<h([1-6])[^>]*>(.*?)<\/h\1>\s*$/i);
    if (htmlHeading) {
      html.push(`<h${htmlHeading[1]}>${renderInlineMarkdown(stripHtml(htmlHeading[2]), currentPath, repoUrl)}</h${htmlHeading[1]}>`);
      index += 1;
      continue;
    }

    const heading = line.match(/^(#{1,6})\s+(.+?)\s*#*\s*$/);
    if (heading) {
      const level = heading[1].length;
      html.push(`<h${level}>${renderInlineMarkdown(heading[2], currentPath, repoUrl)}</h${level}>`);
      index += 1;
      continue;
    }

    if (/^\s*(?:---|\*\*\*|___)\s*$/.test(line)) {
      html.push("<hr />");
      index += 1;
      continue;
    }

    if (isTableStart(lines, index)) {
      const tableLines = [lines[index], lines[index + 1]];
      index += 2;
      while (index < lines.length && /^\s*\|.*\|\s*$/.test(lines[index])) {
        tableLines.push(lines[index]);
        index += 1;
      }
      html.push(renderTable(tableLines, currentPath, repoUrl));
      continue;
    }

    if (/^\s*>\s?/.test(line)) {
      const quoteLines = [];
      while (index < lines.length && /^\s*>\s?/.test(lines[index])) {
        quoteLines.push(lines[index].replace(/^\s*>\s?/, ""));
        index += 1;
      }
      html.push(`<blockquote>${renderMarkdown(quoteLines.join("\n"), currentPath, repoUrl)}</blockquote>`);
      continue;
    }

    if (/^\s*[-*+]\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*[-*+]\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*[-*+]\s+/, ""));
        index += 1;
      }
      html.push(`<ul>${items.map((item) => `<li>${renderInlineMarkdown(item, currentPath, repoUrl)}</li>`).join("")}</ul>`);
      continue;
    }

    if (/^\s*\d+\.\s+/.test(line)) {
      const items = [];
      while (index < lines.length && /^\s*\d+\.\s+/.test(lines[index])) {
        items.push(lines[index].replace(/^\s*\d+\.\s+/, ""));
        index += 1;
      }
      html.push(`<ol>${items.map((item) => `<li>${renderInlineMarkdown(item, currentPath, repoUrl)}</li>`).join("")}</ol>`);
      continue;
    }

    const paragraph = [];
    while (
      index < lines.length &&
      lines[index].trim() &&
      !/^\s*```/.test(lines[index]) &&
      !/^(#{1,6})\s+/.test(lines[index]) &&
      !/^\s*(?:---|\*\*\*|___)\s*$/.test(lines[index]) &&
      !/^\s*>\s?/.test(lines[index]) &&
      !/^\s*[-*+]\s+/.test(lines[index]) &&
      !/^\s*\d+\.\s+/.test(lines[index]) &&
      !isTableStart(lines, index)
    ) {
      paragraph.push(lines[index]);
      index += 1;
    }
    html.push(`<p>${renderInlineMarkdown(paragraph.join(" "), currentPath, repoUrl)}</p>`);
  }

  return html.join("\n");
}

function isTableStart(lines, index) {
  return (
    index + 1 < lines.length &&
    /^\s*\|.*\|\s*$/.test(lines[index]) &&
    /^\s*\|?\s*:?-{3,}:?\s*(?:\|\s*:?-{3,}:?\s*)+\|?\s*$/.test(lines[index + 1])
  );
}

function renderTable(tableLines, currentPath, repoUrl) {
  const rows = tableLines.map(parseTableRow);
  const head = rows[0] || [];
  const body = rows.slice(2);
  return `
    <div class="markdown-table-wrap">
      <table>
        <thead><tr>${head.map((cell) => `<th>${renderInlineMarkdown(cell, currentPath, repoUrl)}</th>`).join("")}</tr></thead>
        <tbody>
          ${body
            .map((row) => `<tr>${row.map((cell) => `<td>${renderInlineMarkdown(cell, currentPath, repoUrl)}</td>`).join("")}</tr>`)
            .join("")}
        </tbody>
      </table>
    </div>
  `;
}

function parseTableRow(line) {
  return line
    .trim()
    .replace(/^\|/, "")
    .replace(/\|$/, "")
    .split("|")
    .map((cell) => cell.trim());
}

function renderInlineMarkdown(value, currentPath, repoUrl) {
  let text = escapeHtml(value);

  text = text.replace(/!\[([^\]]*)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (_, alt, href) => {
    const resolved = resolveDocumentUrl(unescapeHtml(href), currentPath, repoUrl);
    return `<img src="${escapeHtml(resolved)}" alt="${alt}" loading="lazy" />`;
  });

  text = text.replace(/\[([^\]]+)\]\(([^)\s]+)(?:\s+&quot;[^&]*&quot;)?\)/g, (_, label, href) => {
    const resolved = resolveDocumentUrl(unescapeHtml(href), currentPath, repoUrl);
    const externalAttrs = /^https?:\/\//i.test(resolved) ? ' target="_blank" rel="noreferrer"' : "";
    return `<a href="${escapeHtml(resolved)}"${externalAttrs}>${label}</a>`;
  });

  text = text.replace(/`([^`]+)`/g, "<code>$1</code>");
  text = text.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  text = text.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  return text;
}

function resolveDocumentUrl(href, currentPath, repoUrl) {
  const trimmed = href.trim();
  if (!trimmed || /^(?:https?:|mailto:|tel:|#|data:image\/)/i.test(trimmed)) return trimmed;

  const match = trimmed.match(/^([^?#]*)([?#].*)?$/);
  const pathPart = match ? match[1] : trimmed;
  const suffix = match ? match[2] || "" : "";

  if (pathPart.startsWith("/")) return `${pathPart}${suffix}`;

  const currentDir = currentPath.includes("/") ? currentPath.slice(0, currentPath.lastIndexOf("/")) : "";
  const repoTarget = normalizeDocPath(`docs/${currentDir ? `${currentDir}/` : ""}${pathPart}`);

  if (repoTarget.startsWith("docs/")) {
    const docsTarget = repoTarget.slice("docs/".length);
    if (docsTarget.toLowerCase().endsWith(".md")) {
      return `reader.html?path=${encodeURIComponent(docsTarget)}${suffix}`;
    }
    return `${docsTarget}${suffix}`;
  }

  return `${repoUrl}/blob/main/${repoTarget}${suffix}`;
}

function normalizeDocPath(path) {
  const parts = [];
  for (const rawPart of String(path).split("/")) {
    const part = rawPart.trim();
    if (!part || part === ".") continue;
    if (part === "..") {
      if (parts.length && parts[parts.length - 1] !== "..") {
        parts.pop();
      } else {
        parts.push("..");
      }
    } else {
      parts.push(part);
    }
  }
  return parts.join("/");
}

function encodePathForUrl(path) {
  return String(path)
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");
}

function stripHtml(value) {
  return String(value).replace(/<[^>]+>/g, "");
}

function unescapeHtml(value) {
  return String(value)
    .replaceAll("&amp;", "&")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">");
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
