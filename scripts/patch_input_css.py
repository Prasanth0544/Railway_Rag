"""Append modern all-in-one input box CSS to styles.css."""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')

INPUT_CSS = """
/* ── Modern all-in-one input box ─────────────────────────── */
.ask-form {
  padding: 12px 24px 16px;
  border-top: 1px solid var(--border);
  background: var(--bg2);
  flex-shrink: 0;
}

.input-box {
  display: flex;
  flex-direction: column;
  border-radius: 16px;
  border: 1.5px solid var(--border2);
  background: var(--bg3);
  transition: border-color 0.18s, box-shadow 0.18s;
  overflow: hidden;
}

.input-box:focus-within {
  border-color: var(--rail);
  box-shadow: 0 0 0 3px var(--rail-dim);
  background: var(--bg2);
}

/* Textarea inside the box — no separate border */
.input-box textarea {
  width: 100%;
  padding: 14px 16px 6px;
  border: none;
  background: transparent;
  color: var(--ink);
  resize: none;
  outline: none;
  line-height: 1.55;
  font-size: 0.95rem;
  font-family: inherit;
  min-height: 52px;
  max-height: 140px;
}

.input-box textarea::placeholder { color: var(--ink3); }
.input-box textarea:focus { box-shadow: none; }

/* Toolbar row at the bottom of the box */
.input-toolbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 10px 8px;
  gap: 8px;
  border-top: 1px solid var(--border);
}

.input-toolbar-left,
.input-toolbar-right {
  display: flex;
  align-items: center;
  gap: 6px;
}

/* Small icon-only tool buttons */
.tool-btn {
  display: inline-flex;
  align-items: center;
  gap: 5px;
  padding: 5px 9px;
  border-radius: 8px;
  border: 1px solid transparent;
  background: transparent;
  color: var(--ink2);
  font-size: 0.78rem;
  font-family: inherit;
  cursor: pointer;
  transition: background 0.15s, color 0.15s, border-color 0.15s;
}
.tool-btn:hover {
  background: var(--bg2);
  border-color: var(--border);
  color: var(--ink);
}

/* Char count inside box */
.input-box .char-count {
  font-size: 0.72rem;
  color: var(--ink3);
  font-variant-numeric: tabular-nums;
}

/* Shortcut pill inside box */
.input-box .shortcut-hint {
  padding: 2px 6px;
  border-radius: 5px;
  border: 1px solid var(--border);
  background: var(--bg2);
  font-size: 0.68rem;
  color: var(--ink3);
  font-family: inherit;
}

/* Ask button stays red pill */
.input-box .ask-btn {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  padding: 7px 16px;
  border-radius: 10px;
  background: var(--rail);
  color: #fff;
  font-weight: 700;
  font-size: 0.85rem;
  font-family: inherit;
  border: none;
  cursor: pointer;
  transition: background 0.15s, transform 0.1s;
}
.input-box .ask-btn:hover {
  background: #c0392b;
  transform: translateY(-1px);
}
.input-box .ask-btn:disabled {
  opacity: 0.55;
  cursor: not-allowed;
  transform: none;
}

/* File preview inside the box */
.input-box .file-preview {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 14px;
  background: rgba(255,255,255,0.04);
  border-bottom: 1px solid var(--border);
  font-size: 0.8rem;
  gap: 10px;
}

/* Remove old standalone styles that conflict */
"""

css_path = r'c:\Users\prasa\projects\Railway RAG Assistant\web\styles.css'
with open(css_path, 'r', encoding='utf-8') as f:
    css = f.read()

# Override old ask-form, textarea, form-footer, form-actions CSS
# by appending new styles (CSS cascade — last definition wins)
css = css.rstrip() + '\n' + INPUT_CSS

# Bump version in styles.css itself has no version, but bump index.html
with open(css_path, 'w', encoding='utf-8') as f:
    f.write(css)

print('styles.css: input box CSS appended, total:', len(css))
