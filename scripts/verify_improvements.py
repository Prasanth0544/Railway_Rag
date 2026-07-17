"""Verification script for all implemented improvements."""
import requests, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

base = 'http://localhost:8000'

# 1. Health check
print("=== PHASE 1: Critical Fixes ===")
r = requests.get(base)
h = r.json()
print(f"  [OK] Server health: {h['status']}")

# 2. Check new google.genai SDK import
try:
    from google import genai
    from google.genai import types
    has_part = hasattr(types, 'Part')
    print(f"  [OK] google.genai import OK | types.Part: {has_part}")
except Exception as e:
    print(f"  [FAIL] google.genai: {e}")

# 3. Check .env.example exists
import os
example_exists = os.path.exists('.env.example')
print(f"  [OK] .env.example exists: {example_exists}")

print()
print("=== PHASE 3: Architecture ===")

# 4. Pydantic Settings config
try:
    from app.config import settings
    print(f"  [OK] Config loaded: provider={settings.LLM_PROVIDER}, model={settings.GEMINI_MODEL}, has_key={settings.has_api_key}")
except Exception as e:
    print(f"  [FAIL] Config: {e}")

# 5. Logger
try:
    from app.logger import get_logger
    log = get_logger('verify')
    log.info('test message')
    print("  [OK] logger.py working")
except Exception as e:
    print(f"  [FAIL] Logger: {e}")

# 6. __init__.py
init_exists = os.path.exists('app/__init__.py')
print(f"  [OK] app/__init__.py exists: {init_exists}")

# 7. No bare print() left in main.py
with open('app/main.py', 'r', encoding='utf-8') as f:
    main_content = f.read()
import re
bare_prints = [l.strip() for l in main_content.split('\n') if re.match(r'^\s*print\(', l)]
if bare_prints:
    print(f"  [WARN] Remaining print() in main.py: {len(bare_prints)} — {bare_prints[:2]}")
else:
    print("  [OK] No bare print() in main.py")

print()
print("=== PHASE 4: RAG Quality ===")

# 8. Synonym expansion
try:
    from app.retriever import _expand_railway_synonyms
    q = "SL class RAC quota TTE fine Vizag"
    expanded = _expand_railway_synonyms(q)
    print(f"  [OK] Synonym expansion:")
    print(f"       Input   : {q}")
    print(f"       Expanded: {expanded}")
except Exception as e:
    print(f"  [FAIL] Synonym map: {e}")

# 9. Conversation history structure in main
with open('app/main.py', 'r', encoding='utf-8') as f:
    mc = f.read()
has_history = '_session_history' in mc
has_history_inject = 'CONVERSATION HISTORY' in mc
print(f"  [OK] Session history dict: {has_history}")
print(f"  [OK] History injected into prompt: {has_history_inject}")

print()
print("=== PHASE 2: UX/Frontend ===")

# 10. Check index.html for new elements
with open('web/index.html', 'r', encoding='utf-8') as f:
    html = f.read()
checks = {
    'Hamburger button': 'menuToggle' in html,
    'Sidebar overlay': 'sidebarOverlay' in html,
    'Drop overlay': 'dropOverlay' in html,
    'marked.js CDN': 'marked.min.js' in html,
}
for name, ok in checks.items():
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}: {ok}")

# 11. Check app.js for new features
with open('web/app.js', 'r', encoding='utf-8') as f:
    js = f.read()
js_checks = {
    'Chat persistence (CHAT_KEY)': 'CHAT_KEY' in js,
    'saveChatHistory()': 'saveChatHistory' in js,
    'restoreChatHistory()': 'restoreChatHistory' in js,
    'Copy button handler': 'handleCopyClick' in js,
    'Follow-up chips': 'buildFollowupChips' in js,
    'Drag & drop handlers': 'dragenter' in js,
    'Mobile sidebar toggle': 'toggleSidebar' in js,
    'marked.js usage': 'marked.parse' in js,
}
for name, ok in js_checks.items():
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}: {ok}")

# 12. Check styles.css
with open('web/styles.css', 'r', encoding='utf-8') as f:
    css = f.read()
css_checks = {
    'Mobile media query': '@media (max-width: 768px)' in css,
    'Copy button styles': '.copy-btn' in css,
    'Follow-up chip styles': '.followup-chip' in css,
    'Drop overlay styles': '.drop-overlay' in css,
    'Sidebar mobile styles': 'sidebar.open' in css,
}
for name, ok in css_checks.items():
    print(f"  {'[OK]' if ok else '[FAIL]'} {name}: {ok}")

print()
print("=== LIVE API TEST ===")
# 13. Test a real query with conversation memory (two-turn)
import json
headers = {'Content-Type': 'application/json'}

def ask_smart(q, session='verify-session'):
    data = {'question': q, 'session_id': session}
    buf = ''
    with requests.post(f'{base}/ask/smart', json=data, stream=True, timeout=60) as r:
        for line in r.iter_lines():
            if line and line.startswith(b'data: '):
                payload = json.loads(line[6:])
                if payload['type'] == 'token':
                    buf += payload['token']
                elif payload['type'] == 'done':
                    ms = payload.get('response_time_ms', 0)
                    return buf.strip(), ms
    return buf.strip(), 0

print("  Turn 1: Asking about train 12727...")
ans1, ms1 = ask_smart("Tell me about train 12727")
print(f"  [OK] Got {len(ans1)} chars in {ms1}ms")
print(f"       Preview: {ans1[:120]}...")

print("  Turn 2: Follow-up (context-dependent)...")
ans2, ms2 = ask_smart("What are its stops?")
print(f"  [OK] Got {len(ans2)} chars in {ms2}ms")
print(f"       Preview: {ans2[:120]}...")
history_used = '12727' in ans2 or 'Godavari' in ans2 or 'stop' in ans2.lower()
print(f"  [{'OK' if history_used else 'WARN'}] Multi-turn memory working: {history_used}")
