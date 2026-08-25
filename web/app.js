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
const clearCacheBtn   = document.getElementById('clearCache');
const themeToggle     = document.getElementById('themeToggle');
const chipButtons     = document.querySelectorAll('[data-question]');

const STORAGE_KEY = 'railway-rag-api';
const THEME_KEY   = 'railway-rag-theme';
const CHAT_KEY    = 'railway-rag-chat';

// File upload refs
const fileInput    = document.getElementById('fileInput');
const attachBtn    = document.getElementById('attachBtn');
const filePreview  = document.getElementById('filePreview');
const fileNameEl   = document.getElementById('fileName');
const fileSizeEl   = document.getElementById('fileSize');
const fileRemoveEl = document.getElementById('fileRemove');
let attachedFile   = null;

// Mobile sidebar refs
const menuToggle    = document.getElementById('menuToggle');
const sidebar       = document.querySelector('.sidebar');
const sidebarOverlay = document.getElementById('sidebarOverlay');
const dropOverlay   = document.getElementById('dropOverlay');

// ─── Utilities ────────────────────────────────────────────
const getBase = () => apiBaseInput.value.replace(/\/+$/, '');

const esc = (v) => String(v ?? '')
  .replaceAll('&', '&amp;').replaceAll('<', '&lt;')
  .replaceAll('>', '&gt;').replaceAll('"', '&quot;');

/** Render markdown to HTML — uses marked.js if available, falls back to basic */
function renderMarkdown(text) {
  if (typeof marked !== 'undefined') {
    try {
      marked.setOptions({ breaks: true, gfm: true });
      return marked.parse(text);
    } catch (e) { console.warn('marked.js error, using fallback:', e); }
  }
  // Fallback: basic markdown
  return text
    .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.+?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^[*-] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>\n?)+/g, m => `<ul>${m}</ul>`)
    .split(/\n{2,}/).map(p => {
      p = p.trim();
      if (!p) return '';
      if (p.startsWith('<ul>') || p.startsWith('<li>')) return p;
      return `<p>${p.replace(/\n/g, '<br>')}</p>`;
    }).join('');
}

/** Generate follow-up suggestion chips based on answer context */
function buildFollowupChips(answer, sources) {
  const chips = [];
  const hasRoute = sources?.some(s => s.type === 'train_route');
  const hasTrain = sources?.some(s => s.type === 'train' || s.type === 'train_route');
  const hasStation = sources?.some(s => s.type === 'station');
  const hasRule = sources?.some(s => s.type === 'rule');
  const trainNo = sources?.find(s => s.train_no)?.train_no;

  if (hasTrain && trainNo) {
    if (!hasRoute) chips.push(`Route of train ${trainNo}`);
    chips.push(`Live status of ${trainNo}`);
  }
  if (hasStation) {
    const stn = sources.find(s => s.type === 'station');
    if (stn?.station_name) chips.push(`Trains via ${stn.station_name}`);
  }
  if (hasRoute) chips.push('Cancellation charges');
  if (hasRule) chips.push('Luggage rules');
  if (!chips.length) {
    chips.push('Cancellation charges', 'Sleeper luggage limit');
  }
  return chips.slice(0, 3);
}

/**
 * Strip leaked LLM follow-up suggestion text from the answer.
 *
 * The LLM sometimes appends its own "suggested queries" inline at the end
 * of the answer — e.g. "Live status of 12004Cancellation charges".
 * These match the same patterns as our chip labels but concatenated with
 * no separator. We remove them so they don't pollute the chat bubble.
 *
 * @param {string} text  Raw answer text from LLM
 * @returns {{ clean: string, leaked: string[] }}  Cleaned text + any extracted suggestions
 */
// All known suggestion label words (for partial + full matching)
const SUGGESTION_LABELS = [
  'Live status of \\d+',
  'Route of train \\d+',
  'Trains via [\\w\\s]+',
  'Cancellation charges',
  'Luggage rules',
  'Sleeper luggage limit',
  'TTE duties',
  'Tatkal charges',
  'Refund rules',
  'Train \\d+ info',
  'Train info',
];

// Partial-word prefixes to strip during streaming (incomplete tokens)
const SUGGESTION_PREFIXES = [
  'Luggage', 'Cancellation', 'Sleeper', 'Refund', 'Tatkal',
  'Live status', 'Route of', 'Trains via', 'Train info', 'TTE',
];

function cleanAnswerText(text) {
  // 1. Strip full known suggestion labels at end
  const KNOWN = SUGGESTION_LABELS.join('|');
  const LEAK_PATTERN = new RegExp(`(?:\\s*(?:${KNOWN}))+\\s*$`, 'i');

  let working = text;
  const match = working.trimEnd().match(LEAK_PATTERN);
  if (match) {
    const leakedBlock = match[0].trim();
    working = working.slice(0, working.lastIndexOf(leakedBlock)).trimEnd();
    const leaked = leakedBlock
      .split(/(?=[A-Z][a-z])/)
      .map(s => s.trim())
      .filter(s => s.length > 3);
    return { clean: working, leaked };
  }

  // 2. During streaming: strip trailing partial suggestion words
  //    e.g. text ends with "\nLuggage" before " rules" arrives
  const trailingPartial = SUGGESTION_PREFIXES
    .map(p => new RegExp(`\\s*${p}\\s*$`, 'i'))
    .find(re => re.test(working));
  if (trailingPartial) {
    working = working.replace(trailingPartial, '').trimEnd();
  }

  // 3. Line-by-line fallback: strip trailing lines that are ONLY suggestion labels
  //    This catches cases where the regex fails due to concatenation or whitespace
  const SUGGESTION_LINE_RE = new RegExp(
    `^(?:${SUGGESTION_LABELS.join('|')})[\\s,;.]*$`, 'i'
  );
  const lines = working.split('\n');
  while (lines.length && SUGGESTION_LINE_RE.test(lines[lines.length - 1].trim())) {
    lines.pop();
  }
  working = lines.join('\n').trimEnd();

  return { clean: working, leaked: [] };
}

/**
 * Filter chips so we don't suggest the same topic the user just asked about.
 * e.g. if question is "luggage rules", don't show "Luggage rules" chip.
 */
function filterChips(chips, question) {
  if (!question) return chips;
  const q = question.toLowerCase();
  return chips.filter(chip => {
    const c = chip.toLowerCase();
    // Skip if chip keyword is already in the question
    const keyword = c.split(' ').find(w => w.length > 4);
    return !keyword || !q.includes(keyword);
  });
}

/** Copy button handler */
function handleCopyClick(btn) {
  const card = btn.closest('.answer-card');
  const textEl = card?.querySelector('.answer-text');
  if (!textEl) return;
  navigator.clipboard.writeText(textEl.innerText).then(() => {
    btn.innerHTML = '✅ Copied!';
    btn.classList.add('copied');
    setTimeout(() => {
      btn.innerHTML = '📋 Copy';
      btn.classList.remove('copied');
    }, 2000);
  });
}

/** Feedback thumbs-up / thumbs-down handler */
async function handleFeedbackClick(btn, rating) {
  const card = btn.closest('.answer-card');
  if (!card || card.dataset.rated) return;
  card.dataset.rated = rating;

  // Lock both buttons visually
  card.querySelectorAll('.feedback-btn').forEach(b => b.disabled = true);
  btn.classList.add(rating === 'up' ? 'feedback-up--active' : 'feedback-down--active');

  const question      = card.dataset.question || '';
  const answerPreview = card.querySelector('.answer-text')?.innerText?.slice(0, 300) || '';
  const sessionId     = window._sessionId || 'anon';

  // ── Step 1: Save rating IMMEDIATELY so "just leaving" always counts ──────
  try {
    await fetch(`${getBase()}/feedback`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, answer_preview: answerPreview, rating, comment: '', session_id: sessionId }),
    });
  } catch (_) { /* silent fail */ }

  // ── Step 2: Show comment box for optional extra detail ────────────────────
  const bar = btn.closest('.feedback-bar');
  if (bar && !bar.querySelector('.feedback-comment-box')) {
    const box = document.createElement('div');
    box.className = 'feedback-comment-box';
    box.innerHTML = `
      <textarea class="feedback-textarea" maxlength="300"
        placeholder="${rating === 'up' ? '💬 Add a comment? (optional)' : '💬 What went wrong? (optional)'}"></textarea>
      <div class="feedback-comment-actions">
        <button class="feedback-skip-btn" type="button">Skip</button>
        <button class="feedback-submit-btn" type="button">Submit</button>
      </div>`;
    bar.appendChild(box);
    box.querySelector('textarea').focus();

    const submitComment = async (comment) => {
      box.innerHTML = `<span class="feedback-thanks">${rating === 'up' ? '👍 Thanks for the feedback!' : '👎 Thanks — we\'ll improve.'}</span>`;
      if (!comment) return; // rating already saved, skip empty re-submit
      // Save again with comment text (creates a second richer record)
      try {
        await fetch(`${getBase()}/feedback`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ question, answer_preview: answerPreview, rating, comment, session_id: sessionId }),
        });
      } catch (_) { /* silent fail */ }
    };

    box.querySelector('.feedback-submit-btn').addEventListener('click', () => {
      submitComment(box.querySelector('textarea').value.trim());
    });
    box.querySelector('.feedback-skip-btn').addEventListener('click', () => {
      submitComment('');
    });
    box.querySelector('textarea').addEventListener('keydown', e => {
      if (e.key === 'Enter' && e.ctrlKey) submitComment(e.target.value.trim());
    });
  }
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


/** Phase 4B: Build live train progress visualization from answer text */
function buildLiveTrainViz(answerText, sources) {
  // Only show for live status sources
  const liveSource = sources?.find(s => s.type === 'live_status');
  if (!liveSource) return '';

  // Parse passed and upcoming stations from the answer
  const passedMatches = answerText.match(/\[PASSED\]\s*([^\n|]+)/gi) || [];
  const nextMatches = answerText.match(/\[NEXT\]\s*([^\n|]+)/gi) || [];

  // Also try to parse from "Recently Passed" and "Upcoming Stations" sections
  const passed = passedMatches.map(m => {
    const name = m.replace(/\[PASSED\]\s*/i, '').split('|')[0].trim();
    return name;
  });
  const upcoming = nextMatches.map(m => {
    const name = m.replace(/\[NEXT\]\s*/i, '').split('|')[0].trim();
    return name;
  });

  if (passed.length === 0 && upcoming.length === 0) return '';

  // Parse progress percentage
  const progressMatch = answerText.match(/(\d+)%\s*(?:of route|completed|journey)/i);
  const progress = progressMatch ? parseInt(progressMatch[1]) : null;

  // Parse delay
  const delayMatch = answerText.match(/(\d+)\s*(?:MINUTES?|MIN)\s*LATE/i);
  const onTimeMatch = answerText.match(/On\s*Time/i);
  const delayText = delayMatch ? `${delayMatch[1]} min late` : onTimeMatch ? 'On Time' : null;
  const delayClass = delayMatch ? 'delay-late' : 'delay-ontime';

  // Build station dots
  const allStations = [...passed, ...upcoming];
  const currentIdx = passed.length; // current position is between passed and upcoming

  const stationDots = allStations.map((name, i) => {
    const isPassed = i < passed.length;
    const isCurrent = i === passed.length - 1; // last passed station = current
    const dotClass = isPassed ? 'dot-passed' : 'dot-upcoming';
    const currentMarker = isCurrent ? ' dot-current' : '';
    const shortName = name.length > 15 ? name.substring(0, 13) + '…' : name;
    return `<div class="live-station ${dotClass}${currentMarker}">
      <div class="live-dot"></div>
      <div class="live-station-name">${esc(shortName)}</div>
      ${isCurrent ? '<div class="live-here-badge">HERE</div>' : ''}
    </div>`;
  }).join('');

  // Progress bar percentage
  const progressPct = progress !== null ? progress : (passed.length / Math.max(allStations.length, 1) * 100);

  return `
    <div class="live-train-viz">
      <div class="live-viz-header">
        <span class="live-viz-title">🚂 Live Journey Progress</span>
        ${delayText ? `<span class="live-delay-badge ${delayClass}">${esc(delayText)}</span>` : ''}
        ${progress !== null ? `<span class="live-progress-pct">${progress}%</span>` : ''}
      </div>
      <div class="live-progress-bar">
        <div class="live-progress-fill" style="width:${Math.min(progressPct, 100)}%"></div>
      </div>
      <div class="live-stations-timeline">
        ${stationDots}
      </div>
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



/** Right-side sources panel — chips grouped by type, shown inside the answer card */
function buildSourcesPanel(result) {
  const sources = result.sources || [];
  const timeSec = typeof result.response_time_ms === 'number'
                  ? (result.response_time_ms / 1000).toFixed(2) + 's' : null;

  const TYPE_CONFIG = {
    train:       { icon: '&#128641;', label: 'Trains',     cls: 'chip-train' },
    train_route: { icon: '&#128506;', label: 'Routes',     cls: 'chip-route' },
    rule:        { icon: '&#128218;', label: 'Rules',       cls: 'chip-rule'  },
    station:     { icon: '&#127963;', label: 'Stations',   cls: 'chip-station'},
    reference:   { icon: '&#128196;', label: 'References', cls: 'chip-ref'   },
    live_status: { icon: '&#9889;',   label: 'Live API',   cls: 'chip-live'  },
    pnr_status:  { icon: '&#127903;', label: 'PNR',        cls: 'chip-pnr'   },
  };

  // Group by type
  const groups = {};
  sources.forEach(s => {
    const k = s.type || 'reference';
    if (!groups[k]) groups[k] = { type: k, count: 0, items: [] };
    groups[k].count++;
    groups[k].items.push(s);
  });

  if (!Object.keys(groups).length) return '';

  const numDocs = result.num_documents_retrieved ?? sources.length ?? 0;

  const chipRows = Object.values(groups).map(g => {
    const cfg = TYPE_CONFIG[g.type] || { icon: '&#128196;', label: g.type, cls: 'chip-ref' };
    const uid  = 'sp-' + Math.random().toString(36).slice(2, 7);
    const itemList = g.items.map(s => {
      const sc = typeof s.relevance_score === 'number' ? s.relevance_score.toFixed(3) : '-';
      return '<div class="sp-item"><span class="sp-check">&#10003;</span>'
           + '<span class="sp-name">' + esc(sourceTitle(s)) + '</span>'
           + '<span class="sp-score">' + esc(sc) + '</span></div>';
    }).join('');
    return '<div class="sp-group">'
         + '<button class="sp-chip ' + cfg.cls + '" onclick="(function(b){'
         +   'var d=document.getElementById(\'' + uid + '\');'
         +   'var open=d.classList.toggle(\'sp-expanded\');'
         +   'b.classList.toggle(\'sp-chip--active\',open);'
         + '})(this)">'
         +   '<span class="sp-icon">' + cfg.icon + '</span>'
         +   '<span class="sp-label">' + cfg.label + '</span>'
         +   '<span class="sp-count">' + g.count + '</span>'
         + '</button>'
         + '<div class="sp-items" id="' + uid + '">' + itemList + '</div>'
         + '</div>';
  }).join('');

  const footer = timeSec
    ? '<div class="sp-footer">&#8987; ' + timeSec + ' &nbsp;&middot;&nbsp; ' + numDocs + ' docs</div>'
    : '<div class="sp-footer">' + numDocs + ' docs</div>';

  return '<div class="sources-panel">'
       + '<div class="sp-heading">Sources</div>'
       + chipRows
       + footer
       + '</div>';
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
          <div class="typing-stage-indicator">
            <div class="typing-dots"><span></span><span></span><span></span></div>
            <span class="stage-label" id="stageLabel-${id}">🔍 Classifying your question…</span>
          </div>
        </div>
      </div>
    </div>`;
  chatArea.appendChild(el);
  chatArea.scrollTop = chatArea.scrollHeight;

  // Auto-advance stage labels with animation
  const stages = [
    { text: '🔍 Classifying your question…', delay: 0 },
    { text: '📚 Searching knowledge base…', delay: 800 },
    { text: '✨ Generating answer…', delay: 2500 },
  ];
  const labelEl = document.getElementById(`stageLabel-${id}`);
  if (labelEl) {
    stages.forEach(({ text, delay }) => {
      setTimeout(() => {
        if (labelEl && labelEl.isConnected) {
          labelEl.style.opacity = '0';
          setTimeout(() => {
            labelEl.textContent = text;
            labelEl.style.opacity = '1';
          }, 200);
        }
      }, delay);
    });
  }
  return id;
}

/** appendLoading variant for file uploads — shows thumbnail or PDF icon in the user bubble */
function appendLoadingWithFile(question, file, dataUrl) {
  const id = `msg-${Date.now()}`;
  const el = document.createElement('div');
  el.className = 'chat-message';
  el.id = id;

  const isPdf = file.type === 'application/pdf';
  const isImage = file.type.startsWith('image/');

  // Build the file preview inside the user bubble
  const filePreviewHtml = isPdf
    ? `<div style="display:flex;align-items:center;gap:.5rem;margin-top:.5rem;padding:.5rem .75rem;background:rgba(255,255,255,.07);border-radius:.5rem;font-size:.82rem;color:var(--ink2)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="20" height="20"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
        <span>${esc(file.name)}</span>
      </div>`
    : (isImage && dataUrl)
      ? `<img src="${dataUrl}" alt="${esc(file.name)}" style="display:block;max-width:180px;max-height:140px;border-radius:.6rem;margin-top:.5rem;object-fit:cover;border:2px solid rgba(255,255,255,.12)">`
      : '';

  const questionLine = question && question !== 'Analyze this file'
    ? `<div class="question-text" style="margin-bottom:.3rem">${esc(question)}</div>`
    : '';

  el.innerHTML = `
    <div class="question-bubble">
      ${questionLine}
      ${filePreviewHtml}
    </div>
    <div class="answer-bubble loading-bubble">
      <div class="ai-avatar">🚂</div>
      <div class="answer-content">
        <div class="answer-card">
          <div class="typing-dots"><span></span><span></span><span></span></div>
          <span style="color:var(--ink2);font-size:.88rem">Analysing file with Gemini Vision…</span>
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

  const followupChips = buildFollowupChips(result.answer, result.sources);
  const chipsHtml = followupChips.length
    ? `<div class="followup-chips">${followupChips.map(c => `<button class="followup-chip" data-followup="${esc(c)}">${esc(c)}</button>`).join('')}</div>`
    : '';

  el.querySelector('.answer-bubble').outerHTML = `
    <div class="answer-bubble">
      <div class="ai-avatar">🚂</div>
      <div class="answer-content">
        <div class="answer-card answer-card--grid" data-question="${esc(result.question || '')}">
          <button class="copy-btn" onclick="handleCopyClick(this)">📋 Copy</button>
          <div class="answer-main">
            <div class="answer-text">${renderMarkdown(result.answer.trim())}</div>
            ${routeVizHtml}
            ${pnrCardHtml}
            ${chipsHtml}
            <div class="feedback-bar">
              <span class="feedback-bar-label">Was this helpful?</span>
              <div class="feedback-btns">
                <button class="feedback-btn feedback-up" onclick="handleFeedbackClick(this,'up')" title="Good answer">👍</button>
                <button class="feedback-btn feedback-down" onclick="handleFeedbackClick(this,'down')" title="Bad answer">👎</button>
              </div>
            </div>
          </div>
          ${buildSourcesPanel(result)}
        </div>
      </div>
    </div>`;

  // Wire follow-up chip clicks
  el.querySelectorAll('.followup-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      questionInput.value = chip.dataset.followup;
      updateCharCount();
      askForm.dispatchEvent(new Event('submit'));
    });
  });

  chatArea.scrollTop = chatArea.scrollHeight;
  saveChatHistory();
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
    
    // Parse System Information — auto-format per LLM provider
    if (data.llm_provider) {
      const provider = data.llm_provider.toUpperCase();
      const model    = data.llm_model || '';

      // Provider → friendly label + badge CSS class
      const PROVIDER_META = {
        'GEMINI':     { label: 'Gemini',     cls: 'llm-badge-gemini',     icon: '✦' },
        'OPENROUTER': { label: 'OpenRouter', cls: 'llm-badge-openrouter', icon: '⇌' },
        'LMSTUDIO':   { label: 'LM Studio',  cls: 'llm-badge-lmstudio',   icon: '⚙' },
      };
      const meta = PROVIDER_META[provider] || { label: provider, cls: 'llm-badge-gemini', icon: '◉' };

      // Sidebar: "OpenRouter (stealth/ox-alpha)"
      const sysLLM = document.getElementById('sysLLM');
      if (sysLLM) sysLLM.textContent = `${meta.label} (${model})`;

      // RAG flow badge: update text + swap CSS class for colour
      const flowLLM = document.getElementById('flowLLM');
      if (flowLLM) {
        flowLLM.textContent = `${meta.icon} ${model}`;
        // Swap badge class — remove all known classes first
        flowLLM.classList.remove('llm-badge-gemini', 'llm-badge-openrouter', 'llm-badge-lmstudio');
        flowLLM.classList.add(meta.cls);
        flowLLM.title = `Provider: ${meta.label}`;
      }
    }
    if (data.embedding_model) {
      const val = document.getElementById('sysEmbed');
      if (val) val.textContent = data.embedding_model.split(' ')[0]; // keep concise
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

// ── Per-service health panel ────────────────────────────────────────────────
// Maps service key → { dotId, badgeId }
const SERVICE_ELEMENTS = {
  chromadb:   { dot: 'svcDotChroma', badge: 'svcBadgeChroma' },
  gemini_api: { dot: 'svcDotGemini', badge: 'svcBadgeGemini' },
  openrouter: { dot: 'svcDotOR',     badge: 'svcBadgeOR'     },
  rapidapi:   { dot: 'svcDotRapid',  badge: 'svcBadgeRapid'  },
  rag_chain:  { dot: 'svcDotRAG',    badge: 'svcBadgeRAG'    },
  mongodb:    { dot: 'svcDotMongo',  badge: 'svcBadgeMongo'  },
};

// Maps status string → { dotClass, badgeClass, label }
function resolveServiceState(key, status) {
  const OK_STATES   = ['online', 'connected', 'ready', 'configured'];
  const WARN_STATES = ['warming_up', 'not_configured'];
  const ERR_STATES  = ['offline', 'missing_key'];

  // RapidAPI + OpenRouter are optional — "not_configured" is grey not red
  const optional = ['rapidapi', 'openrouter'];

  if (OK_STATES.includes(status)) {
    return { dot: 'svc-ok', badge: 'badge-ok', label: status };
  }
  if (ERR_STATES.includes(status)) {
    // Optional services missing key = amber, not red
    return optional.includes(key)
      ? { dot: 'svc-warn', badge: 'badge-warn', label: status }
      : { dot: 'svc-err',  badge: 'badge-err',  label: status };
  }
  if (WARN_STATES.includes(status)) {
    return { dot: 'svc-warn', badge: 'badge-warn', label: status };
  }
  return { dot: 'svc-off', badge: '', label: status ?? '—' };
}

async function checkApiHealth() {
  const base = getBase();
  try {
    const res  = await fetch(`${base}/api/health`);
    if (!res.ok) return;                          // silent fail — /health covers main status
    const data = await res.json();

    const svcs = data.services || {};
    for (const [key, ids] of Object.entries(SERVICE_ELEMENTS)) {
      const status = svcs[key] ?? null;
      if (!status) continue;
      const state  = resolveServiceState(key, status);
      const dotEl  = document.getElementById(ids.dot);
      const badgeEl = document.getElementById(ids.badge);
      if (dotEl)   dotEl.className   = `svc-dot ${state.dot}`;
      if (badgeEl) {
        badgeEl.textContent = state.label.replace(/_/g, ' ');
        badgeEl.className   = `svc-badge ${state.badge}`;
      }
    }

    // Update overall status label from /api/health message if healthier wording available
    if (data.status === 'healthy') {
      statusLabel.textContent = 'System Operational';
    } else if (data.status === 'degraded') {
      statusLabel.textContent = 'Partially Degraded';
    }
  } catch (_) {
    // /api/health not reachable — dots stay as amber (loading state)
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

        // Phase 4A: Update stage label based on intent
        const stageEl = document.getElementById(`stageLabel-${msgId}`);
        if (stageEl && stageEl.isConnected) {
          const liveUsed = intent === 'LIVE' || intent === 'HYBRID';
          stageEl.style.opacity = '0';
          setTimeout(() => {
            stageEl.textContent = liveUsed ? '🚂 Fetching live train data…' : '✨ Generating answer…';
            stageEl.style.opacity = '1';
          }, 200);
        }
        
        // Build warning banner if present
        const warningHtml = warnings.length 
          ? `<div class="warning-banner">⚠️ ${esc(warnings.join(', '))}</div>`
          : '';

        // Render the shell of the answer card immediately so we can stream into it
        answerContentEl.innerHTML = `
          <div class="answer-card" data-question="${esc(question || '')}">
            <button class="copy-btn" onclick="handleCopyClick(this)">📋 Copy</button>
            ${warningHtml}
            <div class="answer-text"></div>
            <div class="sources-section" style="display:none">
              <div class="sources-heading">Sources (${sources.length})</div>
              <div class="sources-list-container"></div>
            </div>
          </div>`;
        hasInitializedBubble = true;

      } else if (payload.type === 'token') {
        // Token event: append streaming text (bubble already initialized above)
        if (!hasInitializedBubble) {
          // Fallback if sources event didn't fire
          answerContentEl.innerHTML = `
            <div class="answer-card" data-question="${esc(question || '')}">
              <button class="copy-btn" onclick="handleCopyClick(this)">📋 Copy</button>
              <div class="answer-text"></div>
            </div>`;
          hasInitializedBubble = true;
        }
        answerText += payload.token;
        const textEl = answerContentEl.querySelector('.answer-text');
        if (textEl) {
          // Strip any leaked follow-up suggestion text while streaming
          const { clean } = cleanAnswerText(answerText);
          textEl.innerHTML = renderMarkdown(clean.trim());
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

          // Phase 4B: Render live train progress visualization
          const liveVizHtml = buildLiveTrainViz(answerText, sources);
          if (liveVizHtml) {
            const textEl = cardEl.querySelector('.answer-text');
            if (textEl) textEl.insertAdjacentHTML('afterend', liveVizHtml);
          }

          // Remove placeholder sources shell
          const sourcesSection = cardEl.querySelector('.sources-section');
          if (sourcesSection) sourcesSection.remove();

          // Add grid layout and inject sources panel
          cardEl.classList.add('answer-card--grid');
          const panelResult = {
            sources: sources,
            num_documents_retrieved: stats.numDocs,
            avg_score: stats.avgScore,
            response_time_ms: stats.responseTime,
            llm_model: stats.llmModel,
          };
          const panelHtml = buildSourcesPanel(panelResult);
          if (panelHtml) cardEl.insertAdjacentHTML('beforeend', panelHtml);

          // Wrap answer-text in answer-main div if not already
          const textEl2 = cardEl.querySelector('.answer-text');
          if (textEl2 && !textEl2.closest('.answer-main')) {
            const wrapper = document.createElement('div');
            wrapper.className = 'answer-main';
            textEl2.parentNode.insertBefore(wrapper, textEl2);
            wrapper.appendChild(textEl2);
          }

          // Clean leaked suggestion text before building chips
          const { clean: cleanedAnswer, leaked: leakedSuggestions } = cleanAnswerText(answerText);

          // Re-render final answer text without leakage
          const finalTextEl = cardEl.querySelector('.answer-text');
          if (finalTextEl) finalTextEl.innerHTML = renderMarkdown(cleanedAnswer.trim());

          // Ensure data-question is set for feedback handler
          if (!cardEl.dataset.question) cardEl.dataset.question = question || '';

          // Render follow-up chips inside answer-main
          const followupChips = buildFollowupChips(cleanedAnswer, sources);
          const rawChips = [...new Set([...leakedSuggestions, ...followupChips])];
          // Filter out chips that match the current question topic (e.g. "Luggage rules" when asking about luggage)
          const mergedChips = filterChips(rawChips, question).slice(0, 4);
          const mainDiv = cardEl.querySelector('.answer-main') || cardEl;
          if (mergedChips.length) {
            const chipsHtml = `<div class="followup-chips">${mergedChips.map(c => `<button class="followup-chip" data-followup="${esc(c)}">${esc(c)}</button>`).join('')}</div>`;
            mainDiv.insertAdjacentHTML('beforeend', chipsHtml);
            mainDiv.querySelectorAll('.followup-chip').forEach(chip => {
              chip.addEventListener('click', () => {
                questionInput.value = chip.dataset.followup;
                updateCharCount();
                askForm.dispatchEvent(new Event('submit'));
              });
            });
          }

          // Inject feedback bar INSIDE answer-main so it stays in column 1 of the grid
          if (!cardEl.querySelector('.feedback-bar')) {
            const feedbackBar = document.createElement('div');
            feedbackBar.className = 'feedback-bar';
            feedbackBar.innerHTML = `
              <div class="feedback-bar-row">
                <span class="feedback-bar-label">Was this helpful?</span>
                <div class="feedback-btns">
                  <button class="feedback-btn feedback-up" onclick="handleFeedbackClick(this,'up')" title="Good answer">👍</button>
                  <button class="feedback-btn feedback-down" onclick="handleFeedbackClick(this,'down')" title="Bad answer">👎</button>
                </div>
              </div>`;
            // Append inside answer-main — keeps it in the left column of the grid
            mainDiv.appendChild(feedbackBar);
          }
        }
        chatArea.scrollTop = chatArea.scrollHeight;
        saveChatHistory();

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

refreshBtn.addEventListener('click', () => { checkHealth(); checkApiHealth(); });
apiBaseInput.addEventListener('change', () => { checkHealth(); checkApiHealth(); });

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
  localStorage.removeItem(CHAT_KEY);
  questionInput.focus();
});

clearCacheBtn.addEventListener('click', async () => {
  const btn = clearCacheBtn;
  const originalText = btn.innerHTML;
  btn.disabled = true;
  btn.style.opacity = '0.6';
  btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16" style="animation:spin 0.8s linear infinite"><path d="M21 12a9 9 0 1 1-9-9"/></svg> Clearing...`;
  if (!document.getElementById('spin-style')) {
    const s = document.createElement('style');
    s.id = 'spin-style';
    s.textContent = '@keyframes spin{to{transform:rotate(360deg)}}';
    document.head.appendChild(s);
  }
  try {
    const baseUrl = localStorage.getItem(STORAGE_KEY) || window.location.origin;
    const res = await fetch(`${baseUrl}/clear-cache`, { method: 'POST' });
    const data = await res.json();
    // Brief success toast
    btn.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="16" height="16"><polyline points="20 6 9 17 4 12"/></svg> Cleared!`;
    btn.style.color = '#22c55e';
    console.log('[Cache Cleared]', data);
  } catch (err) {
    btn.innerHTML = `❌ Failed`;
    console.error('[Clear Cache Error]', err);
  } finally {
    setTimeout(() => {
      btn.innerHTML = originalText;
      btn.disabled = false;
      btn.style.opacity = '';
      btn.style.color = '';
    }, 1800);
  }
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

  // Read file as dataURL for thumbnail (images only; skip for PDFs to avoid large strings)
  let dataUrl = null;
  if (file.type.startsWith('image/')) {
    dataUrl = await new Promise((resolve) => {
      const reader = new FileReader();
      reader.onload = (e) => resolve(e.target.result);
      reader.onerror = () => resolve(null);
      reader.readAsDataURL(file);
    });
  }

  const msgId = appendLoadingWithFile(question, file, dataUrl);
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
    const uploadFooterResult = {
      sources: data.sources || [],
      num_documents_retrieved: data.num_documents_retrieved || 0,
      avg_score: data.avg_relevance_score || 0,
      response_time_ms: data.response_time_ms || 0,
      llm_model: (data.llm_model || '—') + ' (Multi-Modal)',
    };
    const sourcesHtml = buildSourcesFooter(uploadFooterResult);
    const statsHtml = '';

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

// ─── Mobile Sidebar Toggle ──────────────────────────────────────
function toggleSidebar() {
  sidebar.classList.toggle('open');
  sidebarOverlay.classList.toggle('active');
}
if (menuToggle) {
  menuToggle.addEventListener('click', toggleSidebar);
}
if (sidebarOverlay) {
  sidebarOverlay.addEventListener('click', toggleSidebar);
}

// ─── Drag & Drop File Upload ───────────────────────────────────
const mainPanel = document.querySelector('.main-panel');
let dragCounter = 0;

if (mainPanel && dropOverlay) {
  mainPanel.addEventListener('dragenter', (e) => {
    e.preventDefault();
    dragCounter++;
    dropOverlay.classList.add('active');
  });
  mainPanel.addEventListener('dragleave', (e) => {
    e.preventDefault();
    dragCounter--;
    if (dragCounter <= 0) {
      dragCounter = 0;
      dropOverlay.classList.remove('active');
    }
  });
  mainPanel.addEventListener('dragover', (e) => {
    e.preventDefault();
  });
  mainPanel.addEventListener('drop', (e) => {
    e.preventDefault();
    dragCounter = 0;
    dropOverlay.classList.remove('active');

    const file = e.dataTransfer?.files?.[0];
    if (!file) return;

    const allowed = ['image/png','image/jpeg','image/jpg','image/webp','application/pdf'];
    if (!allowed.includes(file.type)) {
      alert('Unsupported file type. Please drop an image (PNG/JPG/WEBP) or PDF.');
      return;
    }
    if (file.size > 10 * 1024 * 1024) {
      alert('File too large. Maximum size is 10 MB.');
      return;
    }

    showFilePreview(file);
    questionInput.focus();
  });
}

// ─── Chat History Persistence ──────────────────────────────────
function saveChatHistory() {
  try {
    const html = chatArea.innerHTML;
    // Only save if there are actual messages (not just empty state)
    if (html.includes('chat-message')) {
      localStorage.setItem(CHAT_KEY, html);
    }
  } catch (e) {
    console.warn('Could not save chat history:', e);
  }
}

function restoreChatHistory() {
  try {
    const saved = localStorage.getItem(CHAT_KEY);
    if (saved && saved.includes('chat-message')) {
      chatArea.innerHTML = saved;
      // Re-wire follow-up chip clicks on restored messages
      chatArea.querySelectorAll('.followup-chip').forEach(chip => {
        chip.addEventListener('click', () => {
          questionInput.value = chip.dataset.followup;
          updateCharCount();
          askForm.dispatchEvent(new Event('submit'));
        });
      });
    }
  } catch (e) {
    console.warn('Could not restore chat history:', e);
  }
}

// ─── Init ─────────────────────────────────────────────────────
const savedBase  = localStorage.getItem(STORAGE_KEY);
const savedTheme = localStorage.getItem(THEME_KEY);

const isLocal = location.hostname === 'localhost' || location.hostname === '127.0.0.1';

if (savedBase && (isLocal || !savedBase.includes('localhost') && !savedBase.includes('127.0.0.1'))) {
  apiBaseInput.value = savedBase;
} else {
  // Auto-detect: use same origin when deployed (Render/cloud), localhost only for local dev
  apiBaseInput.value = isLocal ? 'http://localhost:8000' : location.origin;
}
if (savedTheme) document.documentElement.dataset.theme = savedTheme;

updateCharCount();
checkHealth();
checkApiHealth();
initVoiceInput();
restoreChatHistory();

// ─── Mobile Sidebar Toggle ─────────────────────────────────────
function openSidebar() {
  sidebar.classList.add('open');
  sidebarOverlay.classList.add('active');
  document.body.style.overflow = 'hidden';
}

function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
  document.body.style.overflow = '';
}

if (menuToggle) {
  menuToggle.addEventListener('click', () => {
    sidebar.classList.contains('open') ? closeSidebar() : openSidebar();
  });
}

if (sidebarOverlay) {
  sidebarOverlay.addEventListener('click', closeSidebar);
}

// Close sidebar on ESC key
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && sidebar.classList.contains('open')) closeSidebar();
});

// Close sidebar when a chip is clicked on mobile (nice UX)
chipButtons.forEach(btn => {
  btn.addEventListener('click', () => {
    if (window.innerWidth <= 768) closeSidebar();
  });
});


// ─── Phase 4C: Auto-suggestions ────────────────────────────

const suggestionsDropdown = document.getElementById('suggestionsDropdown');

// Curated popular queries for instant suggestions
const POPULAR_SUGGESTIONS = [
  'Trains between Vijayawada and Hyderabad',
  'Route of train 12727',
  'Cancellation charges for AC 3 tier',
  'What is the luggage limit for sleeper class?',
  'Running status of 12728',
  'Trains from Chennai to Delhi',
  'What are TTE duties?',
  'What is tatkal booking?',
  'Trains via Rajahmundry',
  'Vijayawada station code',
  'Senior citizen concession in railways',
  'Refund policy for tatkal tickets',
  'What is the fine for ticketless travel?',
  'Classes available in Vande Bharat',
  'Difference between GNWL and PQWL',
  'What is RAC in railway?',
  'Trains from Vizag to Hyderabad',
  'How to book tatkal ticket?',
  'Platform number for 12727 at Vijayawada',
  'Sleeper luggage limit',
];

let suggestionIndex = -1;

function showSuggestions(filter) {
  if (!suggestionsDropdown) return;

  const query = (filter || '').toLowerCase().trim();
  if (query.length < 2) {
    suggestionsDropdown.style.display = 'none';
    return;
  }

  const matches = POPULAR_SUGGESTIONS.filter(s =>
    s.toLowerCase().includes(query)
  ).slice(0, 6);

  if (matches.length === 0) {
    suggestionsDropdown.style.display = 'none';
    return;
  }

  suggestionIndex = -1;
  suggestionsDropdown.innerHTML = matches.map((s, i) => {
    // Highlight matching portion
    const idx = s.toLowerCase().indexOf(query);
    const before = s.substring(0, idx);
    const match = s.substring(idx, idx + query.length);
    const after = s.substring(idx + query.length);
    return `<div class="suggestion-item" data-index="${i}" data-value="${esc(s)}">
      ${esc(before)}<strong>${esc(match)}</strong>${esc(after)}
    </div>`;
  }).join('');

  suggestionsDropdown.style.display = 'block';

  // Wire click handlers
  suggestionsDropdown.querySelectorAll('.suggestion-item').forEach(item => {
    item.addEventListener('mousedown', (e) => {
      e.preventDefault(); // prevent blur from firing first
      questionInput.value = item.dataset.value;
      updateCharCount();
      suggestionsDropdown.style.display = 'none';
      questionInput.focus();
    });
  });
}

function hideSuggestions() {
  if (suggestionsDropdown) {
    suggestionsDropdown.style.display = 'none';
    suggestionIndex = -1;
  }
}

// Input event — filter suggestions as user types
questionInput.addEventListener('input', () => {
  showSuggestions(questionInput.value);
});

// Blur event — hide suggestions (with delay for click handling)
questionInput.addEventListener('blur', () => {
  setTimeout(() => hideSuggestions(), 200);
});

// Keyboard navigation for suggestions
questionInput.addEventListener('keydown', (e) => {
  if (!suggestionsDropdown || suggestionsDropdown.style.display === 'none') return;

  const items = suggestionsDropdown.querySelectorAll('.suggestion-item');
  if (!items.length) return;

  if (e.key === 'ArrowDown') {
    e.preventDefault();
    suggestionIndex = Math.min(suggestionIndex + 1, items.length - 1);
    items.forEach((item, i) => item.classList.toggle('active', i === suggestionIndex));
  } else if (e.key === 'ArrowUp') {
    e.preventDefault();
    suggestionIndex = Math.max(suggestionIndex - 1, 0);
    items.forEach((item, i) => item.classList.toggle('active', i === suggestionIndex));
  } else if (e.key === 'Enter' && suggestionIndex >= 0) {
    e.preventDefault();
    questionInput.value = items[suggestionIndex].dataset.value;
    updateCharCount();
    hideSuggestions();
  } else if (e.key === 'Escape') {
    hideSuggestions();
  }
});
