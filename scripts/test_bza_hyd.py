"""Live test: BZA->HYD query after retriever intersection fix."""
import requests, json, sys
sys.stdout.reconfigure(encoding='utf-8')

def ask(q, session='bza-hyd-test'):
    buf = ''
    num_docs = 0
    with requests.post('http://localhost:8000/ask/smart',
                       json={'question': q, 'session_id': session},
                       stream=True, timeout=90) as r:
        for line in r.iter_lines():
            if not line or not line.startswith(b'data: '):
                continue
            p = json.loads(line[6:])
            if p['type'] == 'meta':
                num_docs = p.get('num_documents_retrieved', 0)
                sources = p.get('sources', [])
                has_both = any(
                    ('BZA' in str(s) or 'Vijayawada' in str(s)) for s in sources
                )
                print(f"  Docs retrieved: {num_docs} | intent: {p.get('intent')}")
            elif p['type'] == 'token':
                buf += p['token']
            elif p['type'] == 'done':
                return buf.strip(), p.get('response_time_ms', 0), num_docs
    return buf.strip(), 0, num_docs

print("=" * 60)
print("TEST 1: Vijayawada → Hyderabad trains")
print("=" * 60)
ans, ms, docs = ask('Vijayawada to Hyderabad trains')
print(f"  Response ({ms}ms, {docs} docs):")
print(ans[:800])

# Check for success indicators
no_info = 'no information' in ans.lower() or 'not available' in ans.lower()
has_trains = any(c.isdigit() for c in ans[:200])  # train numbers
print()
print(f"  [{'FAIL' if no_info else 'OK'}] LLM gave real answer (not 'no info'): {not no_info}")
print(f"  [{'OK' if has_trains else 'WARN'}] Contains train numbers: {has_trains}")

print()
print("=" * 60)
print("TEST 2: Secunderabad to Vijayawada (reverse)")
print("=" * 60)
ans2, ms2, docs2 = ask('trains from Secunderabad to Vijayawada', session='sc-bza-test')
print(f"  Response ({ms2}ms, {docs2} docs):")
print(ans2[:600])
no_info2 = 'no information' in ans2.lower()
print(f"\n  [{'FAIL' if no_info2 else 'OK'}] Real answer: {not no_info2}")
