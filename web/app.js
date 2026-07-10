/* ── app.js — Railway RAG Assistant UI ── */

// ─── DOM refs ─────────────────────────────────────────────
const apiBaseInput    = document.getElementById('apiBase');
const refreshBtn      = document.getElementById('refreshHealth');
const statusDot       = document.getElementById('statusDot');
const statusLabel     = document.getElementById('statusLabel');
const statusMessage   = document.getElementById('statusMessage');
const chatArea        = document.getElementById('chatArea');
const emptyState      = document.getElementById('emptyState');
const askForm         = document.getElementById('askForm');
const questionInput   = document.getElementById('questionInput');
const charCount       = document.getElementById('charCount');
const submitBtn       = document.getElementById('submitQuestion');
const clearBtn        = document.getElementById('clearChat');
const themeToggle     = document.getElementById('themeToggle');
const chipButtons     = document.querySelectorAll('[data-question]');

const STORAGE_KEY = 'railway-rag-api';
const THEME_KEY   = 'railway-rag-theme';

// File upload refs
const fileInput    = document.getElementById('fileInput');
const attachBtn    = document.getElementById('attachBtn');
const filePreview  = document.getElementById('filePreview');
const fileNameEl   = document.getElementById('fileName');
const fileSizeEl   = document.getElementById('fileSize');
const fileRemoveEl = document.getElementById('fileRemove');
let attachedFile   = null;

// ─── Utilities ────────────────────────────────────────────
const getBase = () => apiBaseInput.value.replace(/\/+$/, '');

const esc = (v) => String(v ?? '')
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

/** Convert basic markdown to HTML: **bold**, *italic*, `code`, bullet lists */
function renderMarkdown(text) {
  return text
    // Bold **text**
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    // Italic *text*
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    // Inline code `code`
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    // Bullet lines starting with * or -
    .replace(/^[*-] (.+)$/gm, '<li>$1</li>')
    // Wrap consecutive <li> in <ul>
    .replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
    // Double newline → paragraph break
    .split(/\n{2,}/).map(p => {
      p = p.trim();
      if (!p) return '';
      if (p.startsWith('<ul>') || p.startsWith('<li>')) return p;
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');
}

/** Extract station codes from a route string like "VSKP > BZA > HYB" */
function parseRouteStations(text) {
  const match = text.match(/Station sequence:\s*([A-Z][A-Z0-9 >]+)/);
  if (!match) return null;
  return match[1].split('>').map(s => s.trim()).filter(Boolean);
}

/** Build route visualization HTML */
function buildRouteViz(stations) {
  if (!stations || stations.length < 2) return '';
  const MAX = 30; // cap for display
  const shown = stations.length > MAX
    ? [...stations.slice(0, 12), '…', ...stations.slice(-6)]
    : stations;

  const html = shown.map((code, i) => {
    const isFirst = (i === 0);
    const isLast  = (i === shown.length - 1);
    const cls     = isFirst ? 'first' : isLast ? 'last' : '';
    const arrow   = (i < shown.length - 1) ? '<span class="route-arrow">›</span>' : '';
    if (code === '…') {
      return `<span class="route-station"><span class="station-code" style="color:var(--ink2)">…</span>${arrow}</span>`;
    }
    return `<span class="route-station"><span class="station-code ${cls}">${esc(code)}</span>${arrow}</span>`;
  }).join('');

  return `
    <div class="route-viz">
      <div class="route-label">Route — ${stations.length} stations</div>
      <div class="route-stations">${html}</div>
    </div>`;
}


/** Build ticket status card HTML for PNR status */
function buildTicketCard(s) {
  if (!s || s.type !== 'pnr_status') return '';
  
  const chartBadge = s.chart_prepared
    ? `<span class="pnr-badge chart-yes">Chart Prepared</span>`
    : `<span class="pnr-badge chart-no">Chart Not Prepared</span>`;
    
  const passengerRows = (s.passengers || []).map(p => {
    const isWL = String(p.current_status || '').toLowerCase().includes('w/l') || String(p.current_status || '').toLowerCase().includes('wl');
    const isRAC = String(p.current_status || '').toLowerCase().includes('rac');
    const statusCls = isWL ? 'wl' : isRAC ? 'rac' : 'cnf';
    const seatInfo = p.coach ? `${esc(p.coach)} / ${esc(p.berth)}` : '—';
    return `
      <div class="pnr-passenger-row">
        <span class="pnr-p-no">Passenger ${esc(p.passenger_no)}</span>
        <span class="pnr-p-status booking">${esc(p.booking_status)}</span>
        <span class="pnr-p-status current ${statusCls}">${esc(p.current_status)}</span>
        <span class="pnr-p-seat">${esc(seatInfo)}</span>
      </div>`;
  }).join('');

  return `
    <div class="pnr-card">
      <div class="pnr-header">
        <div class="pnr-title">🎫 Booking Details (PNR: ${esc(s.pnr)})</div>
        ${chartBadge}
      </div>
      <div class="pnr-body">
        <div class="pnr-meta-grid">
          <div><span class="lbl">Train:</span> <strong class="val">${esc(s.train_no)} - ${esc(s.train_name)}</strong></div>
          <div><span class="lbl">Date of Journey:</span> <strong class="val">${esc(s.date_of_journey)}</strong></div>
        </div>
        <div class="pnr-passengers-list">
          <div class="pnr-passenger-header">
            <span>Passenger</span>
            <span>Booking Status</span>
            <span>Current Status</span>
            <span>Coach/Seat</span>
          </div>
          ${passengerRows || '<div style="padding:1rem;text-align:center;color:var(--ink2)">No passenger details found.</div>'}
        </div>
      </div>
    </div>`;
}

const TYPE_LABEL = {
  train:       '🚆 Train',
  train_route: '🗺 Route',
  station:     '🏠 Station',
  rule:        '📋 Rule',
  reference:   '📚 Ref',
  live_status: '🔴 Live',
  pnr_status:  '🎫 PNR Status',
};

function sourceTitle(s) {
  if (s.type === 'pnr_status') {
    return `PNR ${s.pnr} (${s.train_name || s.train_no || 'Ticket'})`.trim();
  }
  if (s.type === 'live_status') {
    const details = [];
    if (s.current_station) details.push(`at ${s.current_station}`);
    if (s.status) details.push(s.status);
    const detailStr = details.length ? ` - ${details.join(', ')}` : '';
    return `${s.train_no ? '#' + s.train_no : 'Train'} ${s.train_name || ''}${detailStr}`.trim();
  }
  if (s.train_no || s.train_name)
    return `${s.train_no ? '#' + s.train_no : ''} ${s.train_name || ''}`.trim();
  if (s.station_name || s.station_code)
    return `${s.station_name || 'Station'} ${s.station_code ? '(' + s.station_code + ')' : ''}`.trim();
  if (s.rule_title) return s.rule_title;
  return s.category || s.type || 'Source';
}

function buildSourceBadges(sources) {
  if (!sources?.length) return '<p style="color:var(--ink2);font-size:.85rem">No sources.</p>';
  return `<div class="sources-grid">` +
    sources.map(s => `
      <div class="source-badge">
        <span class="source-type-pill ${esc(s.type)}">${esc(TYPE_LABEL[s.type] || s.type)}</span>
        <span class="source-name" title="${esc(sourceTitle(s))}">${esc(sourceTitle(s))}</span>
        <span class="source-score">Score: ${typeof s.relevance_score === 'number' ? s.relevance_score.toFixed(3) : '—'}</span>
      </div>`).join('')
  + `</div>`;
}

function buildChecklistSources(sources) {
  if (!sources?.length) {
    return '<p style="color:var(--ink2);font-size:.85rem">No sources retrieved.</p>';
  }

  return `<div class="sources-list">` +
    sources.map(s => {
      const title = sourceTitle(s);
      const label = TYPE_LABEL[s.type] || s.type;
      const score = typeof s.relevance_score === 'number' ? s.relevance_score.toFixed(3) : '—';
      return `
        <div class="source-item">
          <span class="source-check" aria-hidden="true">✓</span>
          <div class="source-info">
            <span class="source-title-text" title="${esc(title)}">${esc(title)}</span>
            <span class="source-type-pill ${esc(s.type)}">${esc(label)}</span>
            <span class="source-meta-text">Relevance: ${esc(score)}</span>
          </div>
        </div>`;
    }).join('')
  + `</div>`;
}

function buildStatsStrip(stats) {
  const timeSec = typeof stats.responseTime === 'number' ? (stats.responseTime / 1000).toFixed(2) + 's' : '—';
  const score = typeof stats.avgScore === 'number' ? stats.avgScore.toFixed(4) : '—';
  return `
    <div class="stats-strip">
      <div class="stat-box">
        <span class="stat-box-label">Retrieved Docs</span>
        <span class="stat-box-val">${esc(stats.numDocs ?? 0)}</span>
      </div>
      <div class="stat-box">
        <span class="stat-box-label">Similarity Score</span>
        <span class="stat-box-val">${esc(score)}</span>
      </div>
      <div class="stat-box">
        <span class="stat-box-label">Response Time</span>
        <span class="stat-box-val">${esc(timeSec)}</span>
      </div>
      <div class="stat-box">
        <span class="stat-box-label">LLM Engine</span>
        <span class="stat-box-val" title="${esc(stats.llmModel)}">${esc(stats.llmModel || '—')}</span>
      </div>
      <div class="stat-box">
        <span class="stat-box-label">Embeddings</span>
        <span class="stat-box-val" title="${esc(stats.embedModel)}">${esc(stats.embedModel || '—')}</span>
      </div>
    </div>`;
}

// ─── Render helpers ───────────────────────────────────────
function hideEmpty() {
  if (emptyState) emptyState.remove();
}

function appendLoading(question) {
  const id = `msg-${Date.now()}`;
  const el = document.createElement('div');
  el.className = 'chat-message';
  el.id = id;
  el.innerHTML = `
    <div class="question-bubble">
      <div class="question-text">${esc(question)}</div>
    </div>
    <div class="answer-bubble loading-bubble">
      <div class="ai-avatar">🚂</div>
      <div class="answer-content">
        <div class="answer-card">
          <div class="typing-dots"><span></span><span></span><span></span></div>
          <span style="color:var(--ink2);font-size:.88rem">Searching 34,000+ documents…</span>
        </div>
      </div>
    </div>`;
  chatArea.appendChild(el);
  chatArea.scrollTop = chatArea.scrollHeight;
  return id;
}

function replaceWithAnswer(msgId, result) {
  const el = document.getElementById(msgId);
  if (!el) return;

  // Detect route docs in sources
  const routeDoc = result.sources?.find(s => s.type === 'train_route');
  let routeVizHtml = '';
  if (routeDoc) {
    // Try to extract from answer text
    const stations = parseRouteStations(result.answer);
    routeVizHtml = buildRouteViz(stations);
  }

  // Detect PNR doc in sources
  const pnrDoc = result.sources?.find(s => s.type === 'pnr_status');
  let pnrCardHtml = '';
  if (pnrDoc) {
    pnrCardHtml = buildTicketCard(pnrDoc);
  }

  el.querySelector('.answer-bubble').outerHTML = `
    <div class="answer-bubble">
      <div class="ai-avatar">🚂</div>
      <div class="answer-content">
        <div class="answer-card">
          <div class="answer-text">${renderMarkdown(result.answer.trim())}</div>
          ${routeVizHtml}
          ${pnrCardHtml}
          <div class="sources-section">
            <div class="sources-heading">Sources (${result.num_documents_retrieved ?? result.sources?.length ?? 0})</div>
            ${buildSourceBadges(result.sources)}
          </div>
        </div>
      </div>
    </div>`;
  chatArea.scrollTop = chatArea.scrollHeight;
}

function replaceWithError(msgId, message) {
  const el = document.getElementById(msgId);
  if (!el) return;
  el.querySelector('.answer-bubble').outerHTML = `
    <div class="answer-bubble">
      <div class="ai-avatar">🚂</div>
      <div class="answer-content">
        <div class="error-card">${esc(message)}</div>
      </div>
    </div>`;
}

// ─── API calls ────────────────────────────────────────────
async function checkHealth() {
  const base = getBase();
  localStorage.setItem(STORAGE_KEY, base);
  statusDot.className = 'status-dot';
  statusLabel.textContent = 'Connecting…';
  statusMessage.textContent = `Reaching ${base}`;
  try {
    const res = await fetch(`${base}/health`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    statusDot.className = 'status-dot ok';
    statusLabel.textContent = 'API Online';
    statusMessage.textContent = data.message ?? 'Ready';
    
    // Parse collection counts dynamically
    if (data.collections) {
      if (data.collections.trains !== undefined) {
        const val = document.getElementById('statTrains');
        if (val) val.textContent = Number(data.collections.trains).toLocaleString();
      }
      if (data.collections.stations !== undefined) {
        const val = document.getElementById('statStations');
        if (val) val.textContent = Number(data.collections.stations).toLocaleString();
      }
      if (data.collections.train_routes !== undefined) {
        const val = document.getElementById('statRoutes');
        if (val) val.textContent = Number(data.collections.train_routes).toLocaleString();
      }
      if (data.collections.railway_rules !== undefined) {
        const val = document.getElementById('statRules');
        if (val) val.textContent = Number(data.collections.railway_rules).toLocaleString();
      }
    }
    
    // Parse System Information details
    if (data.llm_provider) {
      const displayProvider = data.llm_provider.toUpperCase();
      const val = document.getElementById('sysLLM');
      if (val) val.textContent = `${displayProvider} (${data.llm_model})`;
      
      const flowVal = document.getElementById('flowLLM');
      if (flowVal) flowVal.textContent = data.llm_model;
    }
    if (data.embedding_model) {
      const val = document.getElementById('sysEmbed');
      if (val) val.textContent = data.embedding_model.split(' ')[0]; // keep it concise
    }
    if (data.vector_db) {
      const val = document.getElementById('sysVector');
      if (val) val.textContent = data.vector_db;
    }
    if (data.total_documents !== undefined) {
      const val = document.getElementById('sysDocs');
      if (val) val.textContent = Number(data.total_documents).toLocaleString();
    }
  } catch (e) {
    statusDot.className = 'status-dot error';
    statusLabel.textContent = 'API Offline';
    statusMessage.textContent = 'Run: uvicorn app.main:app --reload';
  }
}

async function submitQuestion(question) {
  hideEmpty();
  const msgId = appendLoading(question);
  submitBtn.disabled = true;

  try {
    const response = await fetch(`${getBase()}/ask/smart`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question }),
    });

    if (!response.ok) {
      const data = await response.json().catch(() => ({}));
      throw new Error(data.detail || `HTTP ${response.status}`);
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let answerText = '';
    let sources = [];
    let stats = {};
    let hasInitializedBubble = false;
    let warnings = [];
    let intent = 'STATIC';

    // Locate the message element we just created
    const el = document.getElementById(msgId);
    if (!el) return;

    // Get reference to the answer content container
    const answerContentEl = el.querySelector('.answer-content');

    const processLine = (line) => {
      const cleaned = line.trim();
      if (!cleaned.startsWith('data: ')) return;
      
      const rawJson = cleaned.slice(6);
      let payload;
      try {
        payload = JSON.parse(rawJson);
      } catch (err) {
        console.error("Failed to parse SSE payload:", rawJson, err);
        return;
      }

      if (payload.type === 'meta') {
        // Meta event: received sources, document count, model info
        sources = payload.sources || [];
        warnings = payload.warnings || [];
        intent = payload.intent || 'STATIC';
        stats = {
          numDocs: payload.num_documents_retrieved,
          avgScore: payload.avg_relevance_score,
          llmModel: `${payload.llm_model} (${intent})`,
          embedModel: payload.embedding_model
        };
        
        // Build warning banner if present
        const warningHtml = warnings.length 
          ? `<div class="warning-banner">⚠️ ${esc(warnings.join(', '))}</div>`
          : '';

        // Render the shell of the answer card immediately so we can stream into it
        answerContentEl.innerHTML = `
          <div class="answer-card">
            ${warningHtml}
            <div class="answer-text"></div>
            <div class="sources-section" style="display:none">
              <div class="sources-heading">Sources (${sources.length})</div>
              <div class="sources-list-container"></div>
            </div>
          </div>`;
        hasInitializedBubble = true;

      } else if (payload.type === 'token') {
        // Token event: append streaming text
        if (!hasInitializedBubble) {
          answerContentEl.innerHTML = `
            <div class="answer-card">
              <div class="answer-text"></div>
            </div>`;
          hasInitializedBubble = true;
        }
        answerText += payload.token;
        const textEl = answerContentEl.querySelector('.answer-text');
        if (textEl) {
          textEl.innerHTML = renderMarkdown(answerText.trim());
        }
        chatArea.scrollTop = chatArea.scrollHeight;

      } else if (payload.type === 'done') {
        // Done event: append statistics & finalize sources
        stats.responseTime = payload.response_time_ms;
        
        const cardEl = answerContentEl.querySelector('.answer-card');
        if (cardEl) {
          // Render route viz if applicable
          const routeDoc = sources.find(s => s.type === 'train_route');
          if (routeDoc) {
            const stations = parseRouteStations(answerText);
            const routeVizHtml = buildRouteViz(stations);
            if (routeVizHtml) {
              // Insert route viz after the answer text but before sources
              const textEl = cardEl.querySelector('.answer-text');
              if (textEl) textEl.insertAdjacentHTML('afterend', routeVizHtml);
            }
          }

          // Render PNR status card if applicable
          const pnrDoc = sources.find(s => s.type === 'pnr_status');
          if (pnrDoc) {
            const pnrCardHtml = buildTicketCard(pnrDoc);
            if (pnrCardHtml) {
              const textEl = cardEl.querySelector('.answer-text');
              if (textEl) textEl.insertAdjacentHTML('afterend', pnrCardHtml);
            }
          }

          // Render checklist sources
          const sourcesSection = cardEl.querySelector('.sources-section');
          if (sourcesSection) {
            sourcesSection.style.display = 'block';
            const listContainer = sourcesSection.querySelector('.sources-list-container');
            if (listContainer) listContainer.innerHTML = buildChecklistSources(sources);
          }

          // Render RAG statistics strip
          cardEl.insertAdjacentHTML('beforeend', buildStatsStrip(stats));
        }
        chatArea.scrollTop = chatArea.scrollHeight;

      } else if (payload.type === 'error') {
        throw new Error(payload.message || "An error occurred during streaming.");
      }
    };

    while (true) {
      const { value, done } = await reader.read();
      if (done) {
        // Process any remaining text in buffer
        if (buffer.trim()) {
          processLine(buffer);
        }
        break;
      }

      buffer += decoder.decode(value, { stream: true });
      
      // Normalize CRLF to LF
      buffer = buffer.replace(/\r\n/g, '\n');
      
      const lines = buffer.split('\n');
      // Save last partial line
      buffer = lines.pop() || '';

      for (const line of lines) {
        processLine(line);
      }
    }

  } catch (e) {
    replaceWithError(msgId, `${e.message}. Make sure the API server is running.`);
  } finally {
    submitBtn.disabled = false;
    questionInput.focus();
  }
}

// ─── Event listeners ──────────────────────────────────────
askForm.addEventListener('submit', (e) => {
  e.preventDefault();
  const q = questionInput.value.trim();
  const hasFile = attachedFile !== null;

  // If there's a file attached, use the upload endpoint
  if (hasFile) {
    if (q.length < 1 && !hasFile) return;
    questionInput.value = '';
    updateCharCount();
    submitWithFile(q || 'Analyze this file', attachedFile);
    clearAttachedFile();
    return;
  }

  if (q.length < 3) return;
  questionInput.value = '';
  updateCharCount();
  submitQuestion(q);
});

// Ctrl+Enter shortcut
questionInput.addEventListener('keydown', (e) => {
  if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
    e.preventDefault();
    askForm.dispatchEvent(new Event('submit'));
  }
});

questionInput.addEventListener('input', updateCharCount);

function updateCharCount() {
  const len = questionInput.value.length;
  charCount.textContent = `${len} / 500`;
  charCount.style.color = len > 450 ? 'var(--rail)' : '';
}

refreshBtn.addEventListener('click', checkHealth);
apiBaseInput.addEventListener('change', checkHealth);

clearBtn.addEventListener('click', () => {
  chatArea.innerHTML = `
    <div class="empty-state" id="emptyState">
      <div class="empty-icon">
        <svg viewBox="0 0 64 64" fill="none">
          <rect x="8" y="16" width="48" height="32" rx="8" stroke="currentColor" stroke-width="3"/>
          <circle cx="20" cy="52" r="5" fill="currentColor" opacity="0.5"/>
          <circle cx="44" cy="52" r="5" fill="currentColor" opacity="0.5"/>
          <line x1="8" y1="32" x2="56" y2="32" stroke="currentColor" stroke-width="2.5"/>
          <line x1="32" y1="16" x2="32" y2="32" stroke="currentColor" stroke-width="2.5"/>
        </svg>
      </div>
      <h2>Ask me anything about Indian Railways</h2>
      <p>I can look up train schedules, routes, station info, cancellation policies, luggage rules, and much more.</p>
    </div>`;
  questionInput.value = '';
  updateCharCount();
  clearAttachedFile();
  questionInput.focus();
});

chipButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    questionInput.value = btn.dataset.question;
    updateCharCount();
    questionInput.focus();
  });
});

// ─── File Upload ──────────────────────────────────────────

function formatFileSize(bytes) {
  if (bytes < 1024) return bytes + ' B';
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
  return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}

function showFilePreview(file) {
  attachedFile = file;
  fileNameEl.textContent = file.name;
  fileSizeEl.textContent = formatFileSize(file.size);
  filePreview.style.display = 'flex';
  attachBtn.classList.add('has-file');
  // Make question text not required when file is attached
  questionInput.removeAttribute('required');
}

function clearAttachedFile() {
  attachedFile = null;
  fileInput.value = '';
  filePreview.style.display = 'none';
  attachBtn.classList.remove('has-file');
  questionInput.setAttribute('required', '');
}

attachBtn.addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  if (!file) return;

  // Validate size (10MB max)
  if (file.size > 10 * 1024 * 1024) {
    alert('File too large. Maximum size is 10 MB.');
    fileInput.value = '';
    return;
  }

  showFilePreview(file);
});

fileRemoveEl.addEventListener('click', clearAttachedFile);

// Handle file upload submission
async function submitWithFile(question, file) {
  hideEmpty();

  // Create user question bubble with file indicator
  const msgId = appendLoading(`${question}\n📎 ${file.name}`);
  submitBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('question', question);

    const res = await fetch(`${getBase()}/ask/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || 'Upload failed');
    }

    const data = await res.json();

    // Render answer using existing helpers
    const answerHtml = renderMarkdown(data.answer || 'No response.');
    const sourcesHtml = data.sources?.length
      ? `<div class="sources-heading">SOURCES (${data.sources.length})</div>` + buildChecklistSources(data.sources)
      : '';
    const statsHtml = buildStatsStrip({
      numDocs: data.num_documents_retrieved || 0,
      avgScore: data.avg_relevance_score || 0,
      responseTime: data.response_time_ms || 0,
      llmModel: `${data.llm_model || '—'} (📷 Multi-Modal)`,
      embedModel: data.embedding_model || '—',
    });

    const el = document.getElementById(msgId);
    if (el) {
      const answerContent = el.querySelector('.answer-content');
      answerContent.innerHTML = `
        <div class="answer-card">
          <div class="answer-text">${answerHtml}</div>
          ${sourcesHtml}
          ${statsHtml}
        </div>`;
    }

  } catch (e) {
    replaceWithError(msgId, `${e.message}. Make sure the API server is running.`);
  } finally {
    submitBtn.disabled = false;
    questionInput.focus();
  }
}

// ─── Theme toggle ─────────────────────────────────────────
themeToggle.addEventListener('click', () => {
  const html = document.documentElement;
  const next = html.dataset.theme === 'dark' ? 'light' : 'dark';
  html.dataset.theme = next;
  localStorage.setItem(THEME_KEY, next);
});

// ─── Voice Input (Web Speech API) ─────────────────────────
const voiceBtn = document.getElementById('voiceBtn');
let recognition = null;
let isRecording = false;

function initVoiceInput() {
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  if (!SpeechRecognition) {
    voiceBtn.classList.add('unsupported');
    voiceBtn.title = 'Voice input not supported in this browser';
    voiceBtn.addEventListener('click', () => {
      alert('Voice input is not supported in your browser.\nPlease use Chrome, Edge, or Safari.');
    });
    return;
  }

  recognition = new SpeechRecognition();
  recognition.continuous = false;
  recognition.interimResults = true;
  recognition.lang = 'en-IN';  // Indian English for railway station names
  recognition.maxAlternatives = 1;

  let finalTranscript = '';

  recognition.onstart = () => {
    isRecording = true;
    voiceBtn.classList.add('recording');
    questionInput.placeholder = '🎤 Listening... speak your question';
    finalTranscript = '';
  };

  recognition.onresult = (event) => {
    let interim = '';
    for (let i = event.resultIndex; i < event.results.length; i++) {
      const transcript = event.results[i][0].transcript;
      if (event.results[i].isFinal) {
        finalTranscript += transcript;
      } else {
        interim += transcript;
      }
    }
    // Show live transcription in the textarea
    questionInput.value = finalTranscript + interim;
    updateCharCount();
  };

  recognition.onend = () => {
    isRecording = false;
    voiceBtn.classList.remove('recording');
    questionInput.placeholder = 'Ask about train 12727, cancellation rules, Vijayawada station…';

    if (finalTranscript.trim()) {
      questionInput.value = finalTranscript.trim();
      updateCharCount();
      questionInput.focus();
    }
  };

  recognition.onerror = (event) => {
    isRecording = false;
    voiceBtn.classList.remove('recording');
    questionInput.placeholder = 'Ask about train 12727, cancellation rules, Vijayawada station…';

    if (event.error === 'no-speech') {
      // Silently ignore — user just didn't speak
    } else if (event.error === 'not-allowed') {
      alert('Microphone access denied.\nPlease allow microphone permission in your browser settings.');
    } else {
      console.warn('Speech recognition error:', event.error);
    }
  };

  voiceBtn.addEventListener('click', () => {
    if (isRecording) {
      recognition.stop();
    } else {
      recognition.start();
    }
  });
}

// ─── Init ─────────────────────────────────────────────────
const savedBase  = localStorage.getItem(STORAGE_KEY);
const savedTheme = localStorage.getItem(THEME_KEY);

if (savedBase)  apiBaseInput.value = savedBase;
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

updateCharCount();
checkHealth();
initVoiceInput();
