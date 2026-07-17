"""Safely append right-side panel CSS without any regex deletions."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

PANEL_CSS = """
/* ── 2-column answer card grid ─────────────────────────── */
.answer-card--grid {
  display: grid;
  grid-template-columns: 1fr 220px;
  gap: 0 16px;
  align-items: start;
}
.answer-card--grid .copy-btn {
  grid-column: 1 / -1;
}
.answer-main {
  min-width: 0;
}

/* ── Right-side sources panel ───────────────────────────── */
.sources-panel {
  display: flex;
  flex-direction: column;
  gap: 6px;
  padding: 10px;
  border-radius: 10px;
  background: var(--bg3);
  border: 1px solid var(--border);
  align-self: start;
  min-width: 0;
}
.sp-heading {
  font-size: 0.68rem;
  font-weight: 700;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  color: var(--ink2);
  margin-bottom: 2px;
}
.sp-group { display: flex; flex-direction: column; gap: 0; }
.sp-chip {
  display: flex;
  align-items: center;
  gap: 5px;
  width: 100%;
  padding: 5px 8px;
  border-radius: 8px;
  border: 1px solid var(--border);
  background: var(--bg2);
  cursor: pointer;
  font-size: 0.76rem;
  font-family: inherit;
  color: var(--ink2);
  text-align: left;
  transition: background 0.15s, border-color 0.15s;
}
.sp-chip:hover, .sp-chip--active {
  background: var(--bg3);
  border-color: var(--border2);
  color: var(--ink);
}
.sp-icon  { font-size: 0.85rem; }
.sp-label { flex: 1; font-weight: 500; }
.sp-count {
  font-size: 0.68rem;
  font-weight: 700;
  padding: 1px 5px;
  border-radius: 999px;
  background: var(--bg3);
  border: 1px solid var(--border);
}
.sp-chip.chip-train   .sp-icon { color: var(--src-train);   }
.sp-chip.chip-route   .sp-icon { color: var(--src-route);   }
.sp-chip.chip-rule    .sp-icon { color: var(--src-rule);    }
.sp-chip.chip-station .sp-icon { color: var(--src-station); }
.sp-chip.chip-ref     .sp-icon { color: var(--src-ref);     }
.sp-chip.chip-live    .sp-icon { color: #22c55e; }
.sp-chip.chip-pnr     .sp-icon { color: #f59e0b; }
.sp-items {
  display: none;
  flex-direction: column;
  gap: 1px;
  margin-top: 2px;
  padding: 4px 6px;
  border-radius: 0 0 8px 8px;
  background: var(--bg2);
  border: 1px solid var(--border);
  border-top: none;
}
.sp-items.sp-expanded { display: flex; }
.sp-item {
  display: flex;
  align-items: center;
  gap: 5px;
  font-size: 0.74rem;
  padding: 3px 0;
  border-bottom: 1px solid var(--border);
}
.sp-item:last-child { border-bottom: none; }
.sp-check { color: var(--rail); font-size: 0.78rem; flex-shrink: 0; }
.sp-name  { flex: 1; color: var(--ink); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sp-score { flex-shrink: 0; color: var(--ink3); font-size: 0.7rem; font-variant-numeric: tabular-nums; }
.sp-footer {
  font-size: 0.68rem;
  color: var(--ink3);
  margin-top: 2px;
  padding-top: 6px;
  border-top: 1px solid var(--border);
  text-align: right;
}
@media (max-width: 640px) {
  .answer-card--grid { grid-template-columns: 1fr; }
  .sources-panel { margin-top: 10px; }
}
"""

path = r'c:\Users\prasa\projects\Railway RAG Assistant\web\styles.css'
with open(path, 'r', encoding='utf-8') as f:
    css = f.read()

# Only append — no deletion
css = css.rstrip() + '\n' + PANEL_CSS

with open(path, 'w', encoding='utf-8') as f:
    f.write(css)
print('Appended panel CSS OK, total chars:', len(css))
