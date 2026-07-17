"""
Full rewrite of replaceWithAnswer + streaming done handler.
Layout: answer text LEFT (70%), sources chip panel RIGHT (30%) — inside the answer card.
"""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')

path = r'c:\Users\prasa\projects\Railway RAG Assistant\web\app.js'
with open(path, 'r', encoding='utf-8') as f:
    src = f.read()

# ── 1. Replace buildSourcesFooter with a right-panel chip builder ─────────────
NEW_RIGHT_PANEL = r'''
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
'''

# Replace old buildSourcesFooter
start = src.find('\n/** Collapsible sources')
if start == -1:
    start = src.find('\n/** Compact collapsible')
if start == -1:
    start = src.find('\n/** Google AI Mode style')
if start == -1:
    start = src.find('\n/** Right-side sources panel')

if start == -1:
    print("ERROR: can't find buildSourcesFooter to replace")
    sys.exit(1)

fn_start = src.find('function build', start)
depth, pos = 0, src.find('{', fn_start)
while pos < len(src):
    if src[pos] == '{': depth += 1
    elif src[pos] == '}':
        depth -= 1
        if depth == 0: break
    pos += 1
src = src[:start] + '\n' + NEW_RIGHT_PANEL + src[pos+1:]
print('STEP 1 OK: buildSourcesPanel written')

# ── 2. Fix replaceWithAnswer to use 2-column grid ─────────────────────────────
OLD_BUBBLE = (
    '  el.querySelector(\'.answer-bubble\').outerHTML = `\n'
    '    <div class="answer-bubble">\n'
    '      <div class="ai-avatar">🚂</div>\n'
    '      <div class="answer-content">\n'
    '        <div class="answer-card">\n'
    '          <button class="copy-btn" onclick="handleCopyClick(this)">📋 Copy</button>\n'
    '          <div class="answer-text">${renderMarkdown(result.answer.trim())}</div>\n'
    '          ${routeVizHtml}\n'
    '          ${pnrCardHtml}\n'
    '        </div>\n'
    '        ${buildSourcesFooter(result)}\n'
    '        ${chipsHtml}\n'
    '      </div>\n'
    '    </div>`;'
)

NEW_BUBBLE = (
    '  el.querySelector(\'.answer-bubble\').outerHTML = `\n'
    '    <div class="answer-bubble">\n'
    '      <div class="ai-avatar">🚂</div>\n'
    '      <div class="answer-content">\n'
    '        <div class="answer-card answer-card--grid">\n'
    '          <button class="copy-btn" onclick="handleCopyClick(this)">📋 Copy</button>\n'
    '          <div class="answer-main">\n'
    '            <div class="answer-text">${renderMarkdown(result.answer.trim())}</div>\n'
    '            ${routeVizHtml}\n'
    '            ${pnrCardHtml}\n'
    '            ${chipsHtml}\n'
    '          </div>\n'
    '          ${buildSourcesPanel(result)}\n'
    '        </div>\n'
    '      </div>\n'
    '    </div>`;'
)

if OLD_BUBBLE in src:
    src = src.replace(OLD_BUBBLE, NEW_BUBBLE)
    print('STEP 2 OK: replaceWithAnswer updated')
else:
    print('STEP 2 FAIL: old bubble not found')
    idx = src.find('buildSourcesFooter(result)')
    print(f'  buildSourcesFooter(result) at: {idx}')

# ── 3. Fix streaming done handler ─────────────────────────────────────────────
OLD_DONE = (
    '          // Remove placeholder sources shell\n'
    '          const sourcesSection = cardEl.querySelector(\'.sources-section\');\n'
    '          if (sourcesSection) sourcesSection.remove();\n'
    '\n'
    '          // Inject chip-style source footer after the card\n'
    '          const footerResult = {\n'
    '            sources: sources,\n'
    '            num_documents_retrieved: stats.numDocs,\n'
    '            avg_score: stats.avgScore,\n'
    '            response_time_ms: stats.responseTime,\n'
    '            llm_model: stats.llmModel,\n'
    '          };\n'
    '          cardEl.insertAdjacentHTML(\'afterend\', buildSourcesFooter(footerResult));\n'
    '\n'
    '          // Render follow-up chips after the footer\n'
    '          const followupChips = buildFollowupChips(answerText, sources);\n'
    '          if (followupChips.length) {\n'
    '            const footer = cardEl.nextElementSibling;\n'
    '            const chipsHtml = `<div class="followup-chips">${followupChips.map(c => `<button class="followup-chip" data-followup="${esc(c)}">${esc(c)}</button>`).join(\'\')}' + '</div>`;\n'
    '            if (footer) footer.insertAdjacentHTML(\'afterend\', chipsHtml);\n'
    '            else cardEl.insertAdjacentHTML(\'afterend\', chipsHtml);\n'
    '            answerContentEl.querySelectorAll(\'.followup-chip\').forEach(chip => {\n'
    '              chip.addEventListener(\'click\', () => {\n'
    '                questionInput.value = chip.dataset.followup;\n'
    '                updateCharCount();\n'
    '                askForm.dispatchEvent(new Event(\'submit\'));\n'
    '              });\n'
    '            });\n'
    '          }'
)

NEW_DONE = (
    '          // Remove placeholder sources shell\n'
    '          const sourcesSection = cardEl.querySelector(\'.sources-section\');\n'
    '          if (sourcesSection) sourcesSection.remove();\n'
    '\n'
    '          // Add grid layout and inject sources panel\n'
    '          cardEl.classList.add(\'answer-card--grid\');\n'
    '          const panelResult = {\n'
    '            sources: sources,\n'
    '            num_documents_retrieved: stats.numDocs,\n'
    '            avg_score: stats.avgScore,\n'
    '            response_time_ms: stats.responseTime,\n'
    '            llm_model: stats.llmModel,\n'
    '          };\n'
    '          const panelHtml = buildSourcesPanel(panelResult);\n'
    '          if (panelHtml) cardEl.insertAdjacentHTML(\'beforeend\', panelHtml);\n'
    '\n'
    '          // Wrap answer-text in answer-main div if not already\n'
    '          const textEl2 = cardEl.querySelector(\'.answer-text\');\n'
    '          if (textEl2 && !textEl2.closest(\'.answer-main\')) {\n'
    '            const wrapper = document.createElement(\'div\');\n'
    '            wrapper.className = \'answer-main\';\n'
    '            textEl2.parentNode.insertBefore(wrapper, textEl2);\n'
    '            wrapper.appendChild(textEl2);\n'
    '          }\n'
    '\n'
    '          // Render follow-up chips inside answer-main\n'
    '          const followupChips = buildFollowupChips(answerText, sources);\n'
    '          if (followupChips.length) {\n'
    '            const mainDiv = cardEl.querySelector(\'.answer-main\') || cardEl;\n'
    '            const chipsHtml = `<div class="followup-chips">${followupChips.map(c => `<button class="followup-chip" data-followup="${esc(c)}">${esc(c)}</button>`).join(\'\')}' + '</div>`;\n'
    '            mainDiv.insertAdjacentHTML(\'beforeend\', chipsHtml);\n'
    '            mainDiv.querySelectorAll(\'.followup-chip\').forEach(chip => {\n'
    '              chip.addEventListener(\'click\', () => {\n'
    '                questionInput.value = chip.dataset.followup;\n'
    '                updateCharCount();\n'
    '                askForm.dispatchEvent(new Event(\'submit\'));\n'
    '              });\n'
    '            });\n'
    '          }'
)

if OLD_DONE in src:
    src = src.replace(OLD_DONE, NEW_DONE)
    print('STEP 3 OK: streaming done handler updated')
else:
    print('STEP 3 FAIL: done handler not found')

with open(path, 'w', encoding='utf-8') as f:
    f.write(src)
print('app.js written')
