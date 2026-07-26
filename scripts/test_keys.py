"""
test_keys.py — Comprehensive diagnostic tool to test all Gemini API keys in .env
(Tests both active and commented-out keys)
"""

import os
import sys
import re
import time
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
    sys.stderr.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def load_all_keys_from_env_file() -> list[tuple[str, str, bool]]:
    """
    Parses .env directly to find all GOOGLE_API_KEY* entries,
    including active and commented-out keys.
    Returns list of (var_name, key_value, is_active).
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
    if not os.path.exists(env_path):
        return []

    keys = []
    seen = set()

    pattern = re.compile(r'^\s*(#\s*)?(GOOGLE_API_KEY_\d+|GOOGLE_API_KEY)\s*=\s*(.+)$')

    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line_str = line.strip()
            match = pattern.match(line_str)
            if match:
                is_commented = bool(match.group(1))
                var_name = match.group(2).strip()
                val = match.group(3).strip().strip('"').strip("'")
                
                # Ignore empty or placeholder keys
                if not val or val == "your-gemini-api-key-here":
                    continue
                
                if val not in seen:
                    seen.add(val)
                    keys.append((var_name, val, not is_commented))

    return keys


def test_key(var_name: str, api_key: str, is_active: bool) -> str:
    from langchain_google_genai import GoogleGenerativeAIEmbeddings
    status_prefix = "[ACTIVE]   " if is_active else "[COMMENTED]"
    
    for attempt in range(3):
        try:
            embeddings = GoogleGenerativeAIEmbeddings(
                model="models/gemini-embedding-001",
                google_api_key=api_key
            )
            embeddings.embed_query("test")
            print(f"  {status_prefix} {var_name:20s} -> ✅ WORKING / FRESH QUOTA")
            return "WORKING"
        except Exception as exc:
            err = str(exc)
            if "getaddrinfo failed" in err or "10065" in err or "10054" in err or "Server disconnected" in err:
                if attempt < 2:
                    time.sleep(2)
                    continue
                print(f"  {status_prefix} {var_name:20s} -> 🌐 NETWORK ERROR (No internet/DNS)")
                return "NETWORK_ERROR"
            elif "PerDay" in err or ("1000" in err and "RESOURCE_EXHAUSTED" in err):
                print(f"  {status_prefix} {var_name:20s} -> ❌ EXHAUSTED (Daily limit hit)")
                return "EXHAUSTED"
            elif "PerMinute" in err:
                print(f"  {status_prefix} {var_name:20s} -> ⏳ RATE LIMITED (Minute limit, wait 60s)")
                return "RATE_LIMITED"
            elif "API_KEY_INVALID" in err or "400" in err:
                print(f"  {status_prefix} {var_name:20s} -> ⚠️ INVALID KEY")
                return "INVALID"
            else:
                print(f"  {status_prefix} {var_name:20s} -> ⚠️ ERROR: {err[:60]}...")
                return "ERROR"
    return "ERROR"


def main():
    print("=" * 70)
    print("Gemini API Keys Quota Diagnostic Tool (Active & Commented Keys)")
    print("=" * 70)

    keys = load_all_keys_from_env_file()
    if not keys:
        print("No API keys found in .env!")
        return

    print(f"Found {len(keys)} unique key(s) in .env:\n")
    working_keys = []
    exhausted_keys = []

    for name, key, is_active in keys:
        res = test_key(name, key, is_active)
        if res == "WORKING":
            working_keys.append(name)
        elif res == "EXHAUSTED":
            exhausted_keys.append(name)

    print("\n" + "=" * 70)
    print(f"SUMMARY: {len(working_keys)} working key(s), {len(exhausted_keys)} exhausted key(s) out of {len(keys)} total.")
    if working_keys:
        print(f"Ready to use right now: {', '.join(working_keys)}")
    print("=" * 70)


if __name__ == "__main__":
    main()
