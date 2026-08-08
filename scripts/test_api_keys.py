"""Test ALL API keys (GOOGLE_API_KEY, GOOGLE_API_KEY_1, etc.) from .env"""
import os, sys, time, re
sys.stdout.reconfigure(encoding="utf-8")

ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")

# Read ALL GOOGLE_API_KEY variants (GOOGLE_API_KEY, GOOGLE_API_KEY_1, etc.)
keys = []
with open(ENV_PATH, encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        m = re.match(r"^GOOGLE_API_KEY(?:_\d+)?\s*=\s*(.+)$", line)
        if m:
            key = m.group(1).strip()
            if key and len(key) > 10:
                keys.append(key)

print(f"Found {len(keys)} API keys in .env\n")
print(f"{'#':>3}  {'Key (last 8)':12}  Status")
print("-" * 50)

working = []
from google import genai

for i, key in enumerate(keys, 1):
    short = f"...{key[-8:]}"
    try:
        client = genai.Client(api_key=key)
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=["test embedding"]
        )
        dims = len(result.embeddings[0].values)
        print(f"{i:3}  {short:12}  ✅ WORKING ({dims}-dim)")
        working.append(key)
    except Exception as e:
        err = str(e)
        if "429" in err or "RESOURCE_EXHAUSTED" in err:
            print(f"{i:3}  {short:12}  ❌ QUOTA EXHAUSTED")
        elif "UNAUTHENTICATED" in err or "API_KEY_INVALID" in err or "401" in err:
            print(f"{i:3}  {short:12}  🔑 INVALID")
        else:
            print(f"{i:3}  {short:12}  ⚠️  {err[:50]}")
    time.sleep(0.5)

print(f"\n{'='*50}")
print(f"✅ Working: {len(working)} / {len(keys)}")

if working:
    # Check if keys are from DIFFERENT projects by testing 2 rapid calls
    if len(working) >= 2:
        print(f"\nTesting if keys have SEPARATE quotas (different accounts)...")
        try:
            c1 = genai.Client(api_key=working[0])
            c2 = genai.Client(api_key=working[1])
            # Make 50 rapid calls with key 1
            test_texts = [f"test doc {i}" for i in range(50)]
            c1.models.embed_content(model="models/gemini-embedding-001", contents=test_texts)
            print(f"   Key #1: 50 embeddings ✅")
            # Immediately try key 2
            c2.models.embed_content(model="models/gemini-embedding-001", contents=test_texts)
            print(f"   Key #2: 50 embeddings ✅ (right after key #1)")
            print(f"\n   🎉 Keys have SEPARATE quotas! Round-robin will work!")
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"   Key #2 rate-limited immediately after key #1")
                print(f"\n   ⚠️  Keys share SAME quota (same Google account)")
                print(f"       Round-robin won't help — use 1 key with pacing instead")
            else:
                print(f"   Test inconclusive: {err[:60]}")
