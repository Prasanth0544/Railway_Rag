# coding: utf-8
"""
test_keys.py - Test all 20 Gemini API keys with gemini-3.6-flash
Run:  .venv\\Scripts\\python test_keys.py
"""
import os
import time
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage

load_dotenv()

TOTAL = 20
MODEL = "gemini-3.6-flash"
results = {"ok": [], "exhausted": [], "invalid": [], "error": []}

print("=" * 60)
print(f"  Key Health Check - All {TOTAL} Keys [{MODEL}]")
print("=" * 60)

for i in range(TOTAL):
    key_name = f"GOOGLE_API_KEY_{i:02d}"
    api_key  = os.getenv(key_name, "")
    if not api_key:
        print(f"  [{i:02d}] {key_name:<24} NOT SET")
        results["invalid"].append(i)
        continue
    try:
        llm  = ChatGoogleGenerativeAI(model=MODEL, google_api_key=api_key, temperature=0)
        resp = llm.invoke([HumanMessage(content="Reply only: OK")])
        c = resp.content
        # gemini-3.6-flash returns list of content parts
        if isinstance(c, list):
            ans = (c[0].get("text", str(c[0])) if isinstance(c[0], dict) else str(c[0])).strip()[:12]
        else:
            ans = str(c).strip()[:12]
        print(f"  [{i:02d}] {key_name:<24} OK -> {ans!r}")
        results["ok"].append(i)
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err or "quota" in err.lower():
            print(f"  [{i:02d}] {key_name:<24} QUOTA EXHAUSTED")
            results["exhausted"].append(i)
        elif "403" in err or "API_KEY_INVALID" in err:
            print(f"  [{i:02d}] {key_name:<24} INVALID KEY")
            results["invalid"].append(i)
        else:
            print(f"  [{i:02d}] {key_name:<24} ERROR: {err[:50]}")
            results["error"].append(i)
    time.sleep(0.3)

print()
print("=" * 60)
print("SUMMARY")
print(f"  Working   : {len(results['ok'])}  -> {results['ok']}")
print(f"  Exhausted : {len(results['exhausted'])} -> {results['exhausted']}")
print(f"  Invalid   : {len(results['invalid'])} -> {results['invalid']}")
print(f"  Error     : {len(results['error'])} -> {results['error']}")
print("=" * 60)
if results["ok"]:
    print(f"  First working key: GOOGLE_API_KEY_{results['ok'][0]:02d}")
if results["exhausted"]:
    print(f"  Exhausted keys recover in 24hrs. Rotation skips them automatically.")
