"""Restructure header to show title + subtitle on one row."""
import sys
sys.stdout.reconfigure(encoding='utf-8')

path = r'c:\Users\prasa\projects\Railway RAG Assistant\web\index.html'
with open(path, 'r', encoding='utf-8') as f:
    h = f.read()

old = (
    '          <div>\n'
    '            <p class="header-eyebrow">Ask about trains, routes, stations &amp; rules</p>\n'
    '            <h1 class="header-title">Railway Assistant</h1>\n'
    '          </div>'
)

new = (
    '          <div class="header-title-group">\n'
    '            <h1 class="header-title">Railway Assistant</h1>\n'
    '            <span class="header-subtitle">Ask about trains, routes, stations &amp; rules</span>\n'
    '          </div>'
)

if old in h:
    h = h.replace(old, new)
    print('OK: header restructured')
else:
    print('FAIL: exact block not found — showing current header area:')
    idx = h.find('header-eyebrow')
    print(repr(h[idx-100:idx+200]))

with open(path, 'w', encoding='utf-8') as f:
    f.write(h)
print('index.html written')
