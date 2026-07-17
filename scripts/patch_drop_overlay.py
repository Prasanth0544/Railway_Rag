"""Add compact drop-overlay CSS and bump version."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

DROP_CSS = """
/* ─── DRAG & DROP OVERLAY ─────────────────────────────── */
.drop-overlay {
  /* Hidden by default — only shown when dragging a file */
  display: none;
  position: absolute;
  inset: 0;
  z-index: 100;
  background: rgba(0,0,0,0.55);
  backdrop-filter: blur(4px);
  border-radius: 12px;
  align-items: center;
  justify-content: center;
}
.drop-overlay.active {
  display: flex;
}
.drop-overlay-content {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 10px;
  color: #fff;
  font-size: 1.1rem;
  font-weight: 600;
}
.drop-overlay-content svg {
  opacity: 0.9;
}

"""

# Insert before the ASK FORM section
MARKER = '/* ─── ASK FORM ────────────────────────────────────────── */'
css_path = r'c:\Users\prasa\projects\Railway RAG Assistant\web\styles.css'
with open(css_path, 'r', encoding='utf-8') as f:
    css = f.read()

if MARKER in css:
    css = css.replace(MARKER, DROP_CSS + MARKER)
    print('OK: drop-overlay CSS inserted before ask-form section')
else:
    css = css.rstrip() + '\n' + DROP_CSS
    print('FALLBACK: appended at end')

with open(css_path, 'w', encoding='utf-8') as f:
    f.write(css)
print('styles.css written, total chars:', len(css))

# Bump version
html_path = r'c:\Users\prasa\projects\Railway RAG Assistant\web\index.html'
with open(html_path, 'r', encoding='utf-8') as f:
    html = f.read()

# Bump to v6
for old, new in [('styles.css?v=5','styles.css?v=6'),('app.js?v=5','app.js?v=6'),
                  ('styles.css?v=4','styles.css?v=6'),('app.js?v=4','app.js?v=6')]:
    html = html.replace(old, new)
with open(html_path, 'w', encoding='utf-8') as f:
    f.write(html)
print('version bumped to v6')
