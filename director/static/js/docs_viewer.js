/**
 * Interactive Documentation & Progressive Tutorial Hub (`docs_viewer.js`).
 *
 * Reads live Markdown files from `/api/docs` + `/api/docs/file`, resolves internal
 * `.md` links and `docs/` SVG/image assets (`/api/docs/asset`), renders GFM tables,
 * callouts, disclosures, and Mermaid flowcharts cleanly with zero external CDNs,
 * and drives an accessible right-hand TOC with scrollspy.
 */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

const SVG_NS = ["http:", "", "www.w3.org", "2000", "svg"].join("/");

const viewerState = {
  defaultDoc: "docs/wiki/spatial_director_guide.md",
  groups: [],
  items: [],
  knownPaths: new Set(),
  activePath: "",
  activeTitle: "",
  prevDocPath: "",
  nextDocPath: "",
  toastTimer: 0,
  searchTimer: 0,
  tocObserver: null,
  slugCounts: new Map(),
};

function showToast(msg, durationMs = 2400) {
  const el = $("#doc-toast");
  if (!el) return;
  el.textContent = msg;
  el.classList.remove("hidden");
  if (viewerState.toastTimer) clearTimeout(viewerState.toastTimer);
  viewerState.toastTimer = setTimeout(() => {
    el.classList.add("hidden");
  }, durationMs);
}

async function copyText(text, label = "Copied") {
  try {
    await navigator.clipboard.writeText(text);
    showToast(`${label}: ${text}`);
  } catch {
    showToast(text);
  }
}

function isExternalHref(href) {
  return /^[a-z][a-z0-9+.-]*:/i.test(href);
}

function isSafeExternalUrl(href) {
  const lower = href.trim().toLowerCase();
  return lower.startsWith("https:") || lower.startsWith("http:") || lower.startsWith("mailto:");
}

function normalizePosixParts(baseDirParts, relHref) {
  const stack = [...baseDirParts];
  for (const seg of relHref.split("/")) {
    if (!seg || seg === ".") continue;
    if (seg === "..") {
      if (stack.length === 0) return null;
      stack.pop();
    } else {
      stack.push(seg);
    }
  }
  return stack.join("/");
}

function resolveRepoTarget(currentDocPath, rawHref) {
  const clean = String(rawHref || "").trim();
  if (!clean) return { kind: "noop" };
  if (clean.startsWith("#")) {
    return { kind: "hash", fragment: clean.slice(1) };
  }
  if (isExternalHref(clean)) {
    return isSafeExternalUrl(clean)
      ? { kind: "external", href: clean }
      : { kind: "noop" };
  }
  const hashIdx = clean.indexOf("#");
  const pathPart = hashIdx >= 0 ? clean.slice(0, hashIdx) : clean;
  const fragment = hashIdx >= 0 ? clean.slice(hashIdx + 1) : "";
  const docParts = String(currentDocPath || "").split("/");
  docParts.pop();
  const resolved = normalizePosixParts(docParts, pathPart);
  if (!resolved) {
    return { kind: "source", path: clean };
  }
  const lowerRes = resolved.toLowerCase();
  if (lowerRes.endsWith(".md") && (viewerState.knownPaths.has(resolved) || resolved.startsWith("docs/") || resolved.startsWith("director/") || resolved === "AGENTS.md" || resolved === "VISION.md" || resolved === "README.md" || resolved === "INDEX.md")) {
    return { kind: "doc", path: resolved, fragment };
  }
  if (/\.(svg|png|jpe?g|webp)$/i.test(lowerRes) && resolved.startsWith("docs/")) {
    return { kind: "asset", path: resolved };
  }
  return { kind: "source", path: fragment ? `${resolved}#${fragment}` : resolved };
}

function makeSlug(rawText) {
  const base = String(rawText || "section")
    .toLowerCase()
    .replace(/[`*_~()[\]{}.,:;!?'"/\\<>+]/g, "")
    .trim()
    .replace(/\s+/g, "-")
    .replace(/-+/g, "-") || "section";
  const count = viewerState.slugCounts.get(base) || 0;
  viewerState.slugCounts.set(base, count + 1);
  return count === 0 ? base : `${base}-${count}`;
}

/* -------------------------------------------------------------------------
 * Inline Markdown -> DOM
 * ---------------------------------------------------------------------- */

function appendInline(parent, rawText, docPath) {
  const text = String(rawText || "");
  let i = 0;
  let plainBuf = "";

  const flushPlain = () => {
    if (plainBuf) {
      parent.appendChild(document.createTextNode(plainBuf));
      plainBuf = "";
    }
  };

  while (i < text.length) {
    // Escaped character
    if (text[i] === "\\" && i + 1 < text.length) {
      plainBuf += text[i + 1];
      i += 2;
      continue;
    }

    // Inline code backticks (supports double backticks ``...`` too)
    if (text[i] === "`") {
      const dbl = text.startsWith("``", i);
      const marker = dbl ? "``" : "`";
      const end = text.indexOf(marker, i + marker.length);
      if (end > i) {
        flushPlain();
        const codeEl = document.createElement("code");
        codeEl.className = "inline-code";
        codeEl.textContent = text.slice(i + marker.length, end).trim();
        parent.appendChild(codeEl);
        i = end + marker.length;
        continue;
      }
    }

    // Inline <br> / <br/> and <kbd>...</kbd>
    if (text.startsWith("<br/>", i) || text.startsWith("<br>", i) || text.startsWith("<br />", i)) {
      flushPlain();
      parent.appendChild(document.createElement("br"));
      i = text.indexOf(">", i) + 1;
      continue;
    }
    if (text.startsWith("<kbd>", i)) {
      const kEnd = text.indexOf("</kbd>", i + 5);
      if (kEnd > i) {
        flushPlain();
        const kbd = document.createElement("kbd");
        kbd.textContent = text.slice(i + 5, kEnd);
        parent.appendChild(kbd);
        i = kEnd + 6;
        continue;
      }
    }

    // Inline HTML anchors <a id="..."></a>
    if (text.startsWith("<a ", i)) {
      const closeTag = text.indexOf("</a>", i);
      if (closeTag > i) {
        const chunk = text.slice(i, closeTag + 4);
        const mId = chunk.match(/\bid\s*=\s*["']([^"']+)["']/i);
        if (mId) {
          flushPlain();
          const anchor = document.createElement("span");
          anchor.id = mId[1];
          anchor.className = "md-anchor-target";
          anchor.tabIndex = -1;
          parent.appendChild(anchor);
          i = closeTag + 4;
          continue;
        }
      }
    }

    // Image ![alt](src)
    if (text.startsWith("![", i)) {
      const closeB = text.indexOf("]", i + 2);
      if (closeB > i && text[closeB + 1] === "(") {
        const closeP = findMatchingParen(text, closeB + 1);
        if (closeP > closeB) {
          const alt = text.slice(i + 2, closeB);
          const srcRaw = text.slice(closeB + 2, closeP).trim();
          const info = resolveRepoTarget(docPath, srcRaw);
          if (info.kind === "asset") {
            flushPlain();
            const fig = document.createElement("figure");
            fig.className = "md-figure";
            const img = document.createElement("img");
            img.loading = "lazy";
            img.alt = alt;
            img.src = `/api/docs/asset?path=${encodeURIComponent(info.path)}`;
            fig.appendChild(img);
            if (alt) {
              const cap = document.createElement("figcaption");
              cap.textContent = alt;
              fig.appendChild(cap);
            }
            parent.appendChild(fig);
            i = closeP + 1;
            continue;
          }
        }
      }
    }

    // Link [label](href)
    if (text[i] === "[") {
      const closeB = findMatchingBracket(text, i);
      if (closeB > i && text[closeB + 1] === "(") {
        const closeP = findMatchingParen(text, closeB + 1);
        if (closeP > closeB) {
          const label = text.slice(i + 1, closeB);
          const hrefRaw = text.slice(closeB + 2, closeP).trim();
          const info = resolveRepoTarget(docPath, hrefRaw);
          flushPlain();
          const a = document.createElement("a");
          a.className = "md-link";
          if (info.kind === "hash") {
            a.href = `#${info.fragment}`;
            a.addEventListener("click", (ev) => {
              ev.preventDefault();
              scrollToHeading(info.fragment);
            });
          } else if (info.kind === "doc") {
            const fragPart = info.fragment ? `#${info.fragment}` : "";
            a.href = `?doc=${encodeURIComponent(info.path)}${fragPart}`;
            a.title = info.path;
            a.addEventListener("click", (ev) => {
              ev.preventDefault();
              loadDocument(info.path, { fragment: info.fragment, pushHistory: true });
            });
          } else if (info.kind === "external") {
            a.href = info.href;
            a.target = "_blank";
            a.rel = "noopener noreferrer";
          } else if (info.kind === "asset") {
            a.href = `/api/docs/asset?path=${encodeURIComponent(info.path)}`;
            a.target = "_blank";
            a.rel = "noopener";
          } else if (info.kind === "source") {
            a.href = "#";
            a.className = "md-link md-source-link";
            a.title = `Source reference (${info.path}) — click to copy repo path`;
            a.addEventListener("click", (ev) => {
              ev.preventDefault();
              copyText(info.path, "Copied repo path");
            });
          } else {
            a.href = "#";
          }
          appendInline(a, label, docPath);
          parent.appendChild(a);
          i = closeP + 1;
          continue;
        }
      }
    }

    // Bold **...**
    if (text.startsWith("**", i)) {
      const end = text.indexOf("**", i + 2);
      if (end > i + 2) {
        flushPlain();
        const strong = document.createElement("strong");
        appendInline(strong, text.slice(i + 2, end), docPath);
        parent.appendChild(strong);
        i = end + 2;
        continue;
      }
    }

    // Italic *...* (avoid breaking on spaced multiplication ` * `)
    if (text[i] === "*" && text[i + 1] !== " " && text[i + 1] !== "*") {
      const end = findClosingSingleStar(text, i + 1);
      if (end > i + 1) {
        flushPlain();
        const em = document.createElement("em");
        appendInline(em, text.slice(i + 1, end), docPath);
        parent.appendChild(em);
        i = end + 1;
        continue;
      }
    }

    // Strikethrough ~~...~~
    if (text.startsWith("~~", i)) {
      const end = text.indexOf("~~", i + 2);
      if (end > i + 2) {
        flushPlain();
        const del = document.createElement("del");
        appendInline(del, text.slice(i + 2, end), docPath);
        parent.appendChild(del);
        i = end + 2;
        continue;
      }
    }

    plainBuf += text[i];
    i += 1;
  }

  flushPlain();
}

function findMatchingBracket(text, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < text.length; i++) {
    if (text[i] === "\\" && i + 1 < text.length) {
      i += 1;
      continue;
    }
    if (text[i] === "[") depth += 1;
    else if (text[i] === "]") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function findMatchingParen(text, openIdx) {
  let depth = 0;
  for (let i = openIdx; i < text.length; i++) {
    if (text[i] === "\\" && i + 1 < text.length) {
      i += 1;
      continue;
    }
    if (text[i] === "(") depth += 1;
    else if (text[i] === ")") {
      depth -= 1;
      if (depth === 0) return i;
    }
  }
  return -1;
}

function findClosingSingleStar(text, startIdx) {
  for (let i = startIdx; i < text.length; i++) {
    if (text[i] === "\\" && i + 1 < text.length) {
      i += 1;
      continue;
    }
    if (text[i] === "`") {
      const endTick = text.indexOf("`", i + 1);
      if (endTick > i) {
        i = endTick;
        continue;
      }
    }
    if (text[i] === "*" && text[i - 1] !== " " && text[i + 1] !== "*") {
      return i;
    }
  }
  return -1;
}

/* -------------------------------------------------------------------------
 * Minimal Built-In Mermaid Flowchart -> Crisp SVG Renderer
 * ---------------------------------------------------------------------- */

function parseMermaidNodeToken(tok, nodeMap) {
  const s = tok.trim();
  const m = s.match(/^([A-Za-z0-9_.-]+)\s*(?:\["([^"]*)"\]|\[\(([^)]*)\)\]|\(([^)]*)\)|\[([^\]]*)\])?$/);
  if (!m) return null;
  const id = m[1];
  const rawLabel = m[2] ?? m[3] ?? m[4] ?? m[5] ?? id;
  if (!nodeMap.has(id)) {
    const lines = String(rawLabel)
      .replace(/<br\s*\/?>/gi, "\n")
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    nodeMap.set(id, { id, lines: lines.length ? lines : [id] });
  } else if (m[2] || m[3] || m[4] || m[5]) {
    const lines = String(rawLabel)
      .replace(/<br\s*\/?>/gi, "\n")
      .split("\n")
      .map((l) => l.trim())
      .filter(Boolean);
    if (lines.length) nodeMap.get(id).lines = lines;
  }
  return id;
}

function renderMermaidCard(srcText) {
  const lines = String(srcText || "").replace(/\r\n/g, "\n").split("\n").map((l) => l.trim()).filter(Boolean);
  if (!lines.length || !/^(?:flowchart|graph)\s+(LR|TD|TB)\b/i.test(lines[0])) {
    return null;
  }
  const isHorizontal = /LR/i.test(lines[0]);
  const nodeMap = new Map();
  const edges = [];

  for (let i = 1; i < lines.length; i++) {
    const line = lines[i];
    if (line.startsWith("%%") || /^(?:subgraph|end|classDef|class|style)\b/i.test(line)) continue;
    const edgeMatch = line.match(/^(.+?)\s*(?:-->|-.->|==>)(?:\|"([^"]*)"\|)?\s*(.+?);?$/);
    if (edgeMatch) {
      const fromId = parseMermaidNodeToken(edgeMatch[1], nodeMap);
      const toId = parseMermaidNodeToken(edgeMatch[3], nodeMap);
      if (fromId && toId) {
        edges.push({ from: fromId, to: toId, label: (edgeMatch[2] || "").trim() });
      }
    } else {
      parseMermaidNodeToken(line.replace(/;$/, ""), nodeMap);
    }
  }

  if (!edges.length || nodeMap.size < 2) {
    return null;
  }

  // Topological longest-path layer assignment
  const depth = new Map();
  for (const id of nodeMap.keys()) depth.set(id, 0);
  for (let pass = 0; pass < nodeMap.size; pass++) {
    let changed = false;
    for (const e of edges) {
      const next = (depth.get(e.from) || 0) + 1;
      if (next > (depth.get(e.to) || 0) && next < nodeMap.size) {
        depth.set(e.to, next);
        changed = true;
      }
    }
    if (!changed) break;
  }

  const columns = new Map();
  for (const node of nodeMap.values()) {
    const col = depth.get(node.id) || 0;
    if (!columns.has(col)) columns.set(col, []);
    columns.get(col).push(node);
  }

  const boxW = 210;
  const boxHBase = 48;
  const gapX = 76;
  const gapY = 26;
  const pad = 22;
  const positions = new Map();
  let maxX = 0;
  let maxY = 0;

  const sortedCols = Array.from(columns.keys()).sort((a, b) => a - b);
  for (const colIdx of sortedCols) {
    const list = columns.get(colIdx);
    let offset = pad;
    for (const node of list) {
      const h = Math.max(boxHBase, 22 + node.lines.length * 16);
      const x = isHorizontal ? pad + colIdx * (boxW + gapX) : offset;
      const y = isHorizontal ? offset : pad + colIdx * (92 + gapY);
      positions.set(node.id, { x, y, w: boxW, h, lines: node.lines });
      offset += (isHorizontal ? h : boxW) + gapY;
      maxX = Math.max(maxX, x + boxW);
      maxY = Math.max(maxY, y + h);
    }
  }

  const svgW = maxX + pad;
  const svgH = maxY + pad;
  const card = document.createElement("div");
  card.className = "mermaid-card";
  const head = document.createElement("div");
  head.className = "mermaid-card-head";
  head.textContent = "Workflow Diagram";
  card.appendChild(head);

  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("class", "mermaid-svg");
  svg.setAttribute("viewBox", `0 0 ${svgW} ${svgH}`);

  const defs = document.createElementNS(SVG_NS, "defs");
  const marker = document.createElementNS(SVG_NS, "marker");
  const markerId = `arrow-${Math.random().toString(36).slice(2, 8)}`;
  marker.setAttribute("id", markerId);
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "9");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "6");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("orient", "auto-start-reverse");
  const arrowPath = document.createElementNS(SVG_NS, "path");
  arrowPath.setAttribute("d", "M 0 1.5 L 10 5 L 0 8.5 z");
  arrowPath.setAttribute("fill", "#475569");
  marker.appendChild(arrowPath);
  defs.appendChild(marker);
  svg.appendChild(defs);

  // Edges first
  for (const e of edges) {
    const a = positions.get(e.from);
    const b = positions.get(e.to);
    if (!a || !b) continue;
    const x1 = isHorizontal ? a.x + a.w : a.x + a.w * 0.5;
    const y1 = isHorizontal ? a.y + a.h * 0.5 : a.y + a.h;
    const x2 = isHorizontal ? b.x : b.x + b.w * 0.5;
    const y2 = isHorizontal ? b.y + b.h * 0.5 : b.y;
    const dx = Math.max(24, Math.abs(x2 - x1) * 0.45);
    const dy = Math.max(24, Math.abs(y2 - y1) * 0.45);
    const path = document.createElementNS(SVG_NS, "path");
    const dAttr = isHorizontal
      ? `M ${x1} ${y1} C ${x1 + dx} ${y1}, ${x2 - dx} ${y2}, ${x2} ${y2}`
      : `M ${x1} ${y1} C ${x1} ${y1 + dy}, ${x2} ${y2 - dy}, ${x2} ${y2}`;
    path.setAttribute("d", dAttr);
    path.setAttribute("class", "mermaid-edge-path");
    path.setAttribute("marker-end", `url(#${markerId})`);
    svg.appendChild(path);

    if (e.label) {
      const mx = (x1 + x2) * 0.5;
      const my = (y1 + y2) * 0.5;
      const lw = Math.min(180, Math.max(46, e.label.length * 6.4 + 14));
      const lrect = document.createElementNS(SVG_NS, "rect");
      lrect.setAttribute("x", String(mx - lw * 0.5));
      lrect.setAttribute("y", String(my - 10));
      lrect.setAttribute("width", String(lw));
      lrect.setAttribute("height", "18");
      lrect.setAttribute("rx", "9");
      lrect.setAttribute("class", "mermaid-edge-label-bg");
      svg.appendChild(lrect);

      const ltxt = document.createElementNS(SVG_NS, "text");
      ltxt.setAttribute("x", String(mx));
      ltxt.setAttribute("y", String(my + 2.5));
      ltxt.setAttribute("text-anchor", "middle");
      ltxt.setAttribute("class", "mermaid-edge-label-text");
      ltxt.textContent = e.label;
      svg.appendChild(ltxt);
    }
  }

  // Nodes on top
  for (const [id, box] of positions.entries()) {
    const g = document.createElementNS(SVG_NS, "g");
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", String(box.x));
    rect.setAttribute("y", String(box.y));
    rect.setAttribute("width", String(box.w));
    rect.setAttribute("height", String(box.h));
    rect.setAttribute("rx", "9");
    rect.setAttribute("class", colIsAccent(depth.get(id) || 0) ? "mermaid-node-rect mermaid-node-accent" : "mermaid-node-rect");
    g.appendChild(rect);

    const startY = box.y + (box.h - (box.lines.length - 1) * 15) * 0.5 + 4;
    const textEl = document.createElementNS(SVG_NS, "text");
    textEl.setAttribute("class", "mermaid-node-text");
    box.lines.forEach((line, idx) => {
      const tspan = document.createElementNS(SVG_NS, "tspan");
      tspan.setAttribute("x", String(box.x + box.w * 0.5));
      tspan.setAttribute("y", String(startY + idx * 15));
      tspan.setAttribute("text-anchor", "middle");
      tspan.textContent = line;
      textEl.appendChild(tspan);
    });
    g.appendChild(textEl);
    svg.appendChild(g);
  }

  card.appendChild(svg);
  return card;
}

function colIsAccent(colIdx) {
  return colIdx === 0 || colIdx === 2;
}

/* -------------------------------------------------------------------------
 * Block Markdown -> DOM
 * ---------------------------------------------------------------------- */

function splitTableRow(line) {
  const s = line.trim().replace(/^\|/, "").replace(/\|$/, "");
  const cells = [];
  let cur = "";
  for (let i = 0; i < s.length; i++) {
    if (s[i] === "\\" && s[i + 1] === "|") {
      cur += "|";
      i += 1;
    } else if (s[i] === "|") {
      cells.push(cur.trim());
      cur = "";
    } else {
      cur += s[i];
    }
  }
  cells.push(cur.trim());
  return cells;
}

function parseTableAlignments(sepLine) {
  const t = String(sepLine || "").trim();
  if (!t.includes("-")) return null;
  const cells = splitTableRow(t);
  if (!cells.length || !cells.every((c) => /^:?-{3,}:?$/.test(c))) {
    return null;
  }
  return cells.map((cell) => {
    const left = cell.startsWith(":");
    const right = cell.endsWith(":");
    if (left && right) return "align-center";
    if (right) return "align-right";
    return "";
  });
}

function renderCodeBlock(lang, codeText) {
  if ((lang || "").toLowerCase() === "mermaid") {
    const diagram = renderMermaidCard(codeText);
    if (diagram) return diagram;
  }
  const wrap = document.createElement("div");
  wrap.className = "code-card";
  const head = document.createElement("div");
  head.className = "code-head";
  const lbl = document.createElement("span");
  lbl.textContent = lang || "text";
  const cp = document.createElement("button");
  cp.type = "button";
  cp.className = "code-copy-btn";
  cp.textContent = "Copy";
  cp.addEventListener("click", () => copyText(codeText, "Copied code"));
  head.append(lbl, cp);
  const pre = document.createElement("pre");
  const code = document.createElement("code");
  code.textContent = codeText;
  pre.appendChild(code);
  wrap.append(head, pre);
  return wrap;
}

function renderBlocksInto(container, lines, docPath, tocHeadings) {
  let i = 0;
  let pendingAnchorId = "";

  const isListStart = (ln) => /^\s*(?:[-*+]\s+|\d+\.\s+)/.test(ln);
  const isBlockBoundary = (ln) => {
    const t = ln.trim();
    if (!t) return true;
    if (/^```/.test(t) || /^~~~/.test(t)) return true;
    if (/^#{1,6}\s+/.test(t)) return true;
    if (/^(?:---+|___+|\*\*\*+)$/.test(t)) return true;
    if (t.startsWith(">")) return true;
    if (t.startsWith("<details")) return true;
    if (isListStart(ln)) return true;
    return false;
  };

  while (i < lines.length) {
    const raw = lines[i];
    const trimmed = raw.trim();

    if (!trimmed) {
      i += 1;
      continue;
    }

    // Standalone <a id="..."></a> before a heading
    const anchorOnly = trimmed.match(/^<a\s+id\s*=\s*["']([^"']+)["']\s*>\s*<\/a>$/i);
    if (anchorOnly) {
      pendingAnchorId = anchorOnly[1];
      i += 1;
      continue;
    }

    // Fenced code blocks
    const fenceMatch = trimmed.match(/^(```+|~~~+)\s*([A-Za-z0-9_+-]*)/);
    if (fenceMatch) {
      const fenceChar = fenceMatch[1][0];
      const fenceLen = fenceMatch[1].length;
      const lang = fenceMatch[2] || "";
      i += 1;
      const codeLines = [];
      while (i < lines.length) {
        const closeMatch = lines[i].trim().match(new RegExp(`^${fenceChar}{${fenceLen},}\\s*$`));
        if (closeMatch) {
          i += 1;
          break;
        }
        codeLines.push(lines[i]);
        i += 1;
      }
      container.appendChild(renderCodeBlock(lang, codeLines.join("\n")));
      pendingAnchorId = "";
      continue;
    }

    // HTML <details> disclosure
    if (/^<details\b/i.test(trimmed)) {
      const det = document.createElement("details");
      det.className = "md-details";
      if (/\bopen\b/i.test(trimmed)) det.open = true;
      let sumText = "Details";
      const innerLines = [];
      i += 1;
      while (i < lines.length) {
        const lt = lines[i].trim();
        if (/^<\/details>/i.test(lt)) {
          i += 1;
          break;
        }
        const mSum = lt.match(/^<summary>(.*)<\/summary>$/i);
        if (mSum) {
          sumText = mSum[1];
        } else {
          innerLines.push(lines[i]);
        }
        i += 1;
      }
      const sum = document.createElement("summary");
      appendInline(sum, sumText, docPath);
      const bodyDiv = document.createElement("div");
      bodyDiv.className = "md-details-body";
      renderBlocksInto(bodyDiv, innerLines, docPath, []);
      det.append(sum, bodyDiv);
      container.appendChild(det);
      pendingAnchorId = "";
      continue;
    }

    // Headings
    const hMatch = trimmed.match(/^(#{1,6})\s+(.+)$/);
    if (hMatch) {
      const level = hMatch[1].length;
      const hText = hMatch[2].replace(/\s+#+\s*$/, "").trim();
      const hEl = document.createElement(`h${level}`);
      const slugId = pendingAnchorId || makeSlug(hText);
      pendingAnchorId = "";
      hEl.id = slugId;
      hEl.tabIndex = -1;
      appendInline(hEl, hText, docPath);
      if (level === 2 || level === 3) {
        const a = document.createElement("a");
        a.href = `#${slugId}`;
        a.className = "md-heading-anchor";
        a.textContent = "#";
        a.setAttribute("aria-label", `Link to ${hText}`);
        a.addEventListener("click", (ev) => {
          ev.preventDefault();
          scrollToHeading(slugId);
        });
        hEl.appendChild(a);
        if (tocHeadings) {
          tocHeadings.push({ level, id: slugId, text: hText.replace(/\[([^\]]+)\]\([^)]+\)/g, "$1").replace(/`/g, "") });
        }
      }
      container.appendChild(hEl);
      i += 1;
      continue;
    }

    // Horizontal rule
    if (/^(?:---+|___+|\*\*\*+)$/.test(trimmed)) {
      container.appendChild(document.createElement("hr"));
      i += 1;
      continue;
    }

    // GFM Table
    if (trimmed.includes("|") && i + 1 < lines.length) {
      const aligns = parseTableAlignments(lines[i + 1]);
      if (aligns) {
        const headers = splitTableRow(lines[i]);
        i += 2;
        const rows = [];
        while (i < lines.length && lines[i].trim().includes("|") && !lines[i].trim().startsWith(">")) {
          rows.push(splitTableRow(lines[i]));
          i += 1;
        }
        const wrap = document.createElement("div");
        wrap.className = "table-wrap";
        const tbl = document.createElement("table");
        const thead = document.createElement("thead");
        const thr = document.createElement("tr");
        headers.forEach((hCell, cIdx) => {
          const th = document.createElement("th");
          if (aligns[cIdx]) th.className = aligns[cIdx];
          appendInline(th, hCell, docPath);
          thr.appendChild(th);
        });
        thead.appendChild(thr);
        tbl.appendChild(thead);

        const tbody = document.createElement("tbody");
        for (const rCells of rows) {
          const tr = document.createElement("tr");
          headers.forEach((_, cIdx) => {
            const td = document.createElement("td");
            if (aligns[cIdx]) td.className = aligns[cIdx];
            appendInline(td, rCells[cIdx] ?? "", docPath);
            tr.appendChild(td);
          });
          tbody.appendChild(tr);
        }
        tbl.appendChild(tbody);
        wrap.appendChild(tbl);
        container.appendChild(wrap);
        continue;
      }
    }

    // Blockquote & GitHub Callout Alerts
    if (trimmed.startsWith(">")) {
      const qLines = [];
      while (i < lines.length && (lines[i].trim().startsWith(">") || (lines[i].trim() !== "" && qLines.length > 0 && !isBlockBoundary(lines[i])))) {
        qLines.push(lines[i].replace(/^\s*>\s?/, ""));
        i += 1;
      }
      const alertMatch = (qLines[0] || "").trim().match(/^\[!(NOTE|TIP|IMPORTANT|WARNING|CAUTION)\]\s*(.*)$/i);
      if (alertMatch) {
        const kind = alertMatch[1].toLowerCase();
        const box = document.createElement("div");
        box.className = `callout callout-${kind}`;
        const title = document.createElement("div");
        title.className = "callout-title";
        title.textContent = alertMatch[1].toUpperCase();
        box.appendChild(title);
        const bodyLines = [alertMatch[2], ...qLines.slice(1)].filter((l, idx) => idx > 0 || l.trim() !== "");
        renderBlocksInto(box, bodyLines, docPath, []);
        container.appendChild(box);
      } else {
        const bq = document.createElement("blockquote");
        bq.className = "md-quote";
        renderBlocksInto(bq, qLines, docPath, []);
        container.appendChild(bq);
      }
      continue;
    }

    // Bullet / Numbered List
    if (isListStart(raw)) {
      const { node, nextIdx } = parseListBlock(lines, i, docPath);
      container.appendChild(node);
      i = nextIdx;
      continue;
    }

    // Paragraph (join contiguous non-boundary lines)
    const pLines = [];
    while (i < lines.length && !isBlockBoundary(lines[i])) {
      // Stop before a GFM table starts
      if (lines[i].includes("|") && i + 1 < lines.length && parseTableAlignments(lines[i + 1])) {
        break;
      }
      pLines.push(lines[i].trim());
      i += 1;
    }
    if (pLines.length) {
      const p = document.createElement("p");
      appendInline(p, pLines.join(" "), docPath);
      container.appendChild(p);
    } else {
      i += 1;
    }
  }
}

function parseListBlock(lines, startIdx, docPath) {
  const firstMatch = lines[startIdx].match(/^(\s*)(?:[-*+]|\d+\.)\s+/);
  const baseIndent = firstMatch ? firstMatch[1].length : 0;
  const ordered = /^\s*\d+\.\s+/.test(lines[startIdx]);
  const listEl = document.createElement(ordered ? "ol" : "ul");
  let i = startIdx;

  while (i < lines.length) {
    const ln = lines[i];
    if (!ln.trim()) {
      if (i + 1 < lines.length && /^\s*(?:[-*+]|\d+\.)\s+/.test(lines[i + 1])) {
        const nextIndent = lines[i + 1].match(/^(\s*)/)[1].length;
        if (nextIndent === baseIndent) {
          i += 1;
          continue;
        }
      }
      break;
    }
    const itemMatch = ln.match(/^(\s*)(?:([-*+])|(\d+)\.)\s+(.*)$/);
    if (!itemMatch) break;
    const indent = itemMatch[1].length;
    if (indent < baseIndent) break;
    if (indent > baseIndent) {
      const sub = parseListBlock(lines, i, docPath);
      if (listEl.lastElementChild) {
        listEl.lastElementChild.appendChild(sub.node);
      } else {
        listEl.appendChild(sub.node);
      }
      i = sub.nextIdx;
      continue;
    }
    const itemOrdered = Boolean(itemMatch[3]);
    if (itemOrdered !== ordered) break;

    const li = document.createElement("li");
    let bodyText = itemMatch[4];
    const taskMatch = bodyText.match(/^\[( |x|X)\]\s+(.*)$/);
    if (taskMatch) {
      const cb = document.createElement("input");
      cb.type = "checkbox";
      cb.disabled = true;
      cb.checked = taskMatch[1].toLowerCase() === "x";
      li.appendChild(cb);
      li.appendChild(document.createTextNode(" "));
      bodyText = taskMatch[2];
    }

    const contLines = [bodyText.trim()];
    i += 1;
    while (i < lines.length) {
      const nxt = lines[i];
      if (!nxt.trim()) break;
      const nxtItem = nxt.match(/^(\s*)(?:[-*+]|\d+\.)\s+/);
      if (nxtItem) {
        if (nxtItem[1].length > baseIndent) {
          break;
        }
        break;
      }
      if (/^\s*(?:```|~~~|>)/.test(nxt)) break;
      contLines.push(nxt.trim());
      i += 1;
    }
    appendInline(li, contLines.join(" "), docPath);

    // Nested list directly underneath this item
    while (i < lines.length && /^\s*(?:[-*+]|\d+\.)\s+/.test(lines[i])) {
      const subIndent = lines[i].match(/^(\s*)/)[1].length;
      if (subIndent <= baseIndent) break;
      const sub = parseListBlock(lines, i, docPath);
      li.appendChild(sub.node);
      i = sub.nextIdx;
    }
    listEl.appendChild(li);
  }
  return { node: listEl, nextIdx: i };
}

/* -------------------------------------------------------------------------
 * Navigation, TOC Scrollspy, Search & Boot
 * ---------------------------------------------------------------------- */

function scrollToHeading(id) {
  if (!id) return;
  const target = document.getElementById(id);
  if (!target) return;
  target.scrollIntoView({ behavior: "smooth", block: "start" });
  target.focus({ preventScroll: true });
}

function updateToc(headings) {
  const nav = $("#doc-toc-nav");
  if (!nav) return;
  if (viewerState.tocObserver) {
    viewerState.tocObserver.disconnect();
    viewerState.tocObserver = null;
  }
  nav.replaceChildren();
  for (const h of headings) {
    const a = document.createElement("a");
    a.href = `#${h.id}`;
    a.className = `toc-link toc-level-${h.level}`;
    a.textContent = h.text;
    a.dataset.targetId = h.id;
    a.addEventListener("click", (ev) => {
      ev.preventDefault();
      scrollToHeading(h.id);
    });
    nav.appendChild(a);
  }

  const mainEl = $("#doc-main");
  const links = $$("a.toc-link", nav);
  if (!mainEl || !links.length) return;

  const setActiveToc = (id) => {
    for (const lnk of links) {
      const on = lnk.dataset.targetId === id;
      lnk.classList.toggle("is-active", on);
      lnk.setAttribute("aria-current", on ? "true" : "false");
    }
  };

  if (CSS.supports("scroll-target-group: auto")) {
    mainEl.addEventListener(
      "scrollend",
      () => {
        const cur = nav.querySelector("a:target-current");
        if (cur?.dataset?.targetId) setActiveToc(cur.dataset.targetId);
      },
      { passive: true }
    );
  }

  viewerState.tocObserver = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        if (entry.isIntersecting && entry.target.id) {
          setActiveToc(entry.target.id);
          break;
        }
      }
    },
    { root: mainEl, rootMargin: "0px 0px -75% 0px", threshold: 0 }
  );
  for (const h of headings) {
    const el = document.getElementById(h.id);
    if (el) viewerState.tocObserver.observe(el);
  }
  setActiveToc(headings[0].id);
}

function renderSidebarTree() {
  const tree = $("#doc-tree");
  if (!tree) return;
  tree.replaceChildren();

  for (let gIdx = 0; gIdx < viewerState.groups.length; gIdx++) {
    const grp = viewerState.groups[gIdx];
    const det = document.createElement("details");
    det.className = "doc-group";
    const containsActive = grp.items.some((it) => it.path === viewerState.activePath);
    if (gIdx === 0 || containsActive) {
      det.open = true;
    }

    const sum = document.createElement("summary");
    sum.className = "doc-group-summary";
    const titleSpan = document.createElement("span");
    titleSpan.textContent = grp.group;
    const countSpan = document.createElement("span");
    countSpan.className = "doc-group-count";
    countSpan.textContent = String(grp.items.length);
    sum.append(titleSpan, countSpan);

    const list = document.createElement("div");
    list.className = "doc-group-list";
    for (const item of grp.items) {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = item.path === viewerState.activePath ? "doc-nav-btn active" : "doc-nav-btn";
      btn.dataset.docPath = item.path;
      const tEl = document.createElement("span");
      tEl.className = "doc-nav-title";
      tEl.textContent = item.title;
      const sEl = document.createElement("span");
      sEl.className = "doc-nav-sub";
      sEl.textContent = item.path;
      btn.append(tEl, sEl);
      btn.addEventListener("click", () => {
        closeMobileNav();
        loadDocument(item.path, { pushHistory: true });
      });
      list.appendChild(btn);
    }

    det.append(sum, list);
    tree.appendChild(det);
  }
}

function updatePagerButtons() {
  const prevBtn = $("#btn-prev-doc");
  const nextBtn = $("#btn-next-doc");
  const idx = viewerState.items.findIndex((it) => it.path === viewerState.activePath);
  if (idx > 0) {
    const prev = viewerState.items[idx - 1];
    viewerState.prevDocPath = prev.path;
    if (prevBtn) {
      prevBtn.disabled = false;
      prevBtn.textContent = `← ${prev.title}`;
    }
  } else {
    viewerState.prevDocPath = "";
    if (prevBtn) {
      prevBtn.disabled = true;
      prevBtn.textContent = "← Previous";
    }
  }
  if (idx >= 0 && idx + 1 < viewerState.items.length) {
    const nxt = viewerState.items[idx + 1];
    viewerState.nextDocPath = nxt.path;
    if (nextBtn) {
      nextBtn.disabled = false;
      nextBtn.textContent = `${nxt.title} →`;
    }
  } else {
    viewerState.nextDocPath = "";
    if (nextBtn) {
      nextBtn.disabled = true;
      nextBtn.textContent = "Next →";
    }
  }
}

async function loadDocument(path, opts = {}) {
  const targetPath = String(path || viewerState.defaultDoc);
  const article = $("#doc-article");
  if (!article) return;

  try {
    const resp = await fetch(`/api/docs/file?path=${encodeURIComponent(targetPath)}`, { cache: "no-store" });
    const data = await resp.json();
    if (!resp.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${resp.status}`);
    }

    viewerState.activePath = data.path;
    viewerState.activeTitle = data.title;
    viewerState.slugCounts = new Map();
    document.title = `${data.title} — H3 Docs`;

    const pathPill = $("#doc-path-pill");
    const grpBadge = $("#doc-group-badge");
    const updMeta = $("#doc-updated-meta");
    if (pathPill) pathPill.textContent = data.path;
    if (grpBadge) grpBadge.textContent = data.group || "Docs";
    if (updMeta && data.mtime) {
      const dt = new Date(Number(data.mtime) * 1000);
      updMeta.textContent = `${(Number(data.bytes || 0) / 1024).toFixed(1)} KB · Updated ${dt.toLocaleTimeString()}`;
    }

    article.replaceChildren();
    const tocHeadings = [];
    const lines = String(data.markdown || "").replace(/\r\n/g, "\n").split("\n");
    renderBlocksInto(article, lines, data.path, tocHeadings);
    updateToc(tocHeadings);
    renderSidebarTree();
    updatePagerButtons();

    if (opts.pushHistory) {
      const frag = opts.fragment ? `#${opts.fragment}` : "";
      const nextUrl = `${window.location.pathname}?doc=${encodeURIComponent(data.path)}${frag}`;
      window.history.pushState({ doc: data.path, fragment: opts.fragment || "" }, "", nextUrl);
    }

    const mainEl = $("#doc-main");
    if (opts.fragment) {
      requestAnimationFrame(() => scrollToHeading(opts.fragment));
    } else if (mainEl) {
      mainEl.scrollTop = 0;
    }
  } catch (err) {
    article.replaceChildren();
    const msg = document.createElement("p");
    msg.textContent = `Could not load document (${targetPath}): ${err.message}`;
    article.appendChild(msg);
  }
}

function appendHighlightedSnippet(parent, text, query) {
  const terms = query.trim().split(/\s+/).filter(Boolean);
  if (!terms.length) {
    parent.textContent = text;
    return;
  }
  const escaped = terms.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const re = new RegExp(`(${escaped.join("|")})`, "ig");
  let last = 0;
  let m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      parent.appendChild(document.createTextNode(text.slice(last, m.index)));
    }
    const mark = document.createElement("mark");
    mark.textContent = m[0];
    parent.appendChild(mark);
    last = m.index + m[0].length;
    if (m[0].length === 0) break;
  }
  if (last < text.length) {
    parent.appendChild(document.createTextNode(text.slice(last)));
  }
}

async function runDocSearch(query) {
  const q = String(query || "").trim();
  const clearBtn = $("#btn-search-clear");
  const tree = $("#doc-tree");
  const resultsBox = $("#doc-search-results");
  clearBtn?.classList.toggle("hidden", !q);
  if (!tree || !resultsBox) return;

  if (!q) {
    resultsBox.classList.add("hidden");
    resultsBox.replaceChildren();
    tree.classList.remove("hidden");
    return;
  }

  tree.classList.add("hidden");
  resultsBox.classList.remove("hidden");
  resultsBox.replaceChildren();
  const loading = document.createElement("div");
  loading.className = "doc-search-status";
  loading.textContent = "Searching…";
  resultsBox.appendChild(loading);

  try {
    const resp = await fetch(`/api/docs?q=${encodeURIComponent(q)}`, { cache: "no-store" });
    const data = await resp.json();
    resultsBox.replaceChildren();
    const hits = Array.isArray(data.results) ? data.results : [];
    const status = document.createElement("div");
    status.className = "doc-search-status";
    status.textContent = hits.length ? `${hits.length} matching doc(s)` : "No matches found.";
    resultsBox.appendChild(status);

    for (const hit of hits) {
      const card = document.createElement("button");
      card.type = "button";
      card.className = "doc-search-card";
      const title = document.createElement("span");
      title.className = "doc-nav-title";
      appendHighlightedSnippet(title, hit.title, q);
      const sub = document.createElement("span");
      sub.className = "doc-nav-sub";
      sub.textContent = `${hit.path} · ${hit.group}`;
      card.append(title, sub);
      for (const snip of hit.snippets || []) {
        const sDiv = document.createElement("div");
        sDiv.className = "doc-search-snippet";
        appendHighlightedSnippet(sDiv, snip, q);
        card.appendChild(sDiv);
      }
      card.addEventListener("click", () => {
        closeMobileNav();
        loadDocument(hit.path, { pushHistory: true });
      });
      resultsBox.appendChild(card);
    }
  } catch (err) {
    resultsBox.replaceChildren();
    const errDiv = document.createElement("div");
    errDiv.className = "doc-search-status";
    errDiv.textContent = `Search failed: ${err.message}`;
    resultsBox.appendChild(errDiv);
  }
}

function openMobileNav() {
  const dlg = $("#mobile-nav-dialog");
  const slot = $("#mobile-nav-slot");
  const host = $("#doc-nav-host");
  if (!dlg || !slot || !host) return;
  slot.appendChild(host);
  if (typeof dlg.showModal === "function") dlg.showModal();
  else dlg.setAttribute("open", "");
}

function closeMobileNav() {
  const dlg = $("#mobile-nav-dialog");
  const sidebar = $("#doc-sidebar");
  const host = $("#doc-nav-host");
  if (sidebar && host && host.parentElement !== sidebar) {
    sidebar.appendChild(host);
  }
  if (dlg?.open && typeof dlg.close === "function") {
    dlg.close();
  }
}

async function initDocsHub() {
  const searchIn = $("#doc-search-input");
  const clearBtn = $("#btn-search-clear");
  const reloadBtn = $("#btn-doc-reload");
  const copyPathBtn = $("#btn-copy-path");

  try {
    const resp = await fetch("/api/docs", { cache: "no-store" });
    const catalog = await resp.json();
    if (!catalog.ok) throw new Error(catalog.error || "Failed to load docs catalog");
    viewerState.defaultDoc = catalog.default_doc || viewerState.defaultDoc;
    viewerState.groups = Array.isArray(catalog.groups) ? catalog.groups : [];
    viewerState.items = Array.isArray(catalog.items) ? catalog.items : [];
    viewerState.knownPaths = new Set(viewerState.items.map((it) => it.path));
    const pill = $("#docs-count-pill");
    if (pill) pill.textContent = `${viewerState.items.length} Markdown docs`;
  } catch (err) {
    showToast(`Failed to index docs: ${err.message}`);
  }

  const params = new URLSearchParams(window.location.search);
  const startDoc = params.get("doc") || viewerState.defaultDoc;
  const startFrag = window.location.hash ? window.location.hash.slice(1) : "";
  await loadDocument(startDoc, { fragment: startFrag, pushHistory: false });

  searchIn?.addEventListener("input", () => {
    if (viewerState.searchTimer) clearTimeout(viewerState.searchTimer);
    viewerState.searchTimer = setTimeout(() => {
      runDocSearch(searchIn.value);
    }, 150);
  });

  clearBtn?.addEventListener("click", () => {
    if (searchIn) {
      searchIn.value = "";
      searchIn.focus();
    }
    runDocSearch("");
  });

  reloadBtn?.addEventListener("click", () => {
    loadDocument(viewerState.activePath || viewerState.defaultDoc, { pushHistory: false });
    showToast("Reloaded document from disk");
  });

  copyPathBtn?.addEventListener("click", () => {
    copyText(viewerState.activePath || viewerState.defaultDoc, "Copied doc path");
  });

  $("#btn-prev-doc")?.addEventListener("click", () => {
    if (viewerState.prevDocPath) {
      loadDocument(viewerState.prevDocPath, { pushHistory: true });
    }
  });
  $("#btn-next-doc")?.addEventListener("click", () => {
    if (viewerState.nextDocPath) {
      loadDocument(viewerState.nextDocPath, { pushHistory: true });
    }
  });

  $("#btn-mobile-nav")?.addEventListener("click", openMobileNav);
  $("#btn-mobile-nav-close")?.addEventListener("click", closeMobileNav);
  $("#mobile-nav-dialog")?.addEventListener("close", closeMobileNav);
  $("#mobile-nav-dialog")?.addEventListener("click", (ev) => {
    if (ev.target === ev.currentTarget) closeMobileNav();
  });

  window.addEventListener("popstate", () => {
    const p = new URLSearchParams(window.location.search);
    const d = p.get("doc") || viewerState.defaultDoc;
    const f = window.location.hash ? window.location.hash.slice(1) : "";
    loadDocument(d, { fragment: f, pushHistory: false });
  });

  window.addEventListener("keydown", (ev) => {
    const tag = (ev.target?.tagName || "").toUpperCase();
    if ((ev.key === "/" && tag !== "INPUT" && tag !== "TEXTAREA") || ((ev.metaKey || ev.ctrlKey) && ev.key.toLowerCase() === "k")) {
      ev.preventDefault();
      searchIn?.focus();
      searchIn?.select();
    } else if (ev.key === "Escape" && searchIn && document.activeElement === searchIn && searchIn.value) {
      searchIn.value = "";
      runDocSearch("");
    }
  });
}

document.addEventListener("DOMContentLoaded", initDocsHub);
