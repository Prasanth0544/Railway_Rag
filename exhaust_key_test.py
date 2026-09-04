"""
exhaust_key_test.py  --  Railway RAG Key Exhaustion Tester
Fire 60 unique railway questions to exhaust key_00 and watch key rotation.
Run:  .venv\\Scripts\\python exhaust_key_test.py
"""
import requests, json, time

BASE       = "http://127.0.0.1:8000"
SESSION_ID = "exhaust-test-session"
PAUSE_SEC  = 1.2

QUESTIONS = [
    "What are the stops of Rajdhani Express from Delhi to Mumbai?",
    "Tell me the route of 12658 Chennai Mail",
    "What is the departure time of Shatabdi from Bangalore to Chennai?",
    "How many coaches does 22691 Rajdhani have?",
    "What is the luggage allowance in Sleeper class?",
    "What are the cancellation charges if I cancel 48 hours before?",
    "Explain the TDR refund policy for delayed trains",
    "What is the tatkal quota booking window?",
    "Tell me about trains from Vijayawada to Hyderabad",
    "What is the seat layout of 3A class coaches?",
    "How many seats are in a Sleeper class coach?",
    "What is GNWL vs PQWL in waiting list?",
    "What is the senior citizen concession on Indian Railways?",
    "Tell me the station code of Secunderabad Junction",
    "What are the trains running between Delhi and Kolkata?",
    "What is premium tatkal quota different from normal tatkal?",
    "Explain RAC ticket rules - do I get a full berth?",
    "What are the TTE duties on overnight trains?",
    "What is the penalty for travelling without ticket in India?",
    "Tell me about the Konkan Railway route and stops",
    "What is the distance between Vijayawada and Delhi by train?",
    "What trains run on Vijayawada to Nizamabad route?",
    "List the trains that pass through Rajahmundry station",
    "Tell me about train 17016 Visakha Express stops",
    "What is the route of 12723 Telangana Express?",
    "What are the charges for excess luggage on Indian Railways?",
    "What is the student concession on Indian Railways?",
    "Explain the Divyaang concession on train tickets",
    "What is 1A class on Indian Railways and how many berths?",
    "What is the difference between Express and Superfast trains?",
    "What are the railway zones of India?",
    "Tell me about South Central Railway division headquarters",
    "What is the food policy for Rajdhani Express passengers?",
    "Can I carry a bicycle on Indian Railways?",
    "Tell me about the Vande Bharat Express speed and seats",
    "Explain the flexi fare scheme in Rajdhani trains",
    "Train classes in Indian Railways from cheapest to expensive?",
    "What is the booking window for general quota tickets?",
    "Can foreign nationals book Indian Railways tickets online?",
    "What are the rules for carrying pets on Indian trains?",
    "What is the emergency quota on Indian Railways?",
    "What is the Ladies quota in train reservations?",
    "What is the rule for travelling with children below 5?",
    "How many kg luggage can I carry in 2A class?",
    "Trains from Hyderabad to Tirupati and their departure times",
    "What is the station code of Nizamabad railway station?",
    "Tell me about train 17001 Hyderabad Warangal Intercity",
    "What are the stops of 12603 Hyderabad Chennai Mail?",
    "Can I travel from Vijayawada to Guntur by local train?",
    "What is the distance from Warangal to Hyderabad by train?",
    "Tell me about train 12728 Godavari Express route and timings",
    "Trains from Rajahmundry to Visakhapatnam with timing",
    "What time does Garib Rath reach Nizamabad from Hyderabad?",
    "What is the Tatkal charge for 2A class from Hyderabad to Delhi?",
    "What documents are required for a RAC passenger?",
    "Can I change my name on a confirmed railway ticket?",
    "What is the minimum age for senior citizen concession?",
    "Explain the difference between waiting list and RAC tickets",
    "What is Platform ticket cost at major railway stations?",
    "What is the policy for unattended luggage on Indian Railways?",
]


def check_key_status():
    try:
        r = requests.get(f"{BASE}/api/key-status", timeout=5)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def ask_question(question):
    try:
        r = requests.post(
            f"{BASE}/ask/smart",
            json={"question": question, "session_id": SESSION_ID},
            timeout=60,
            stream=True,
        )
        preview, exhausted = "", False
        for raw_line in r.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            try:
                ev = json.loads(line[6:])
                t = ev.get("type", "")
                if t == "token":
                    preview += ev.get("token", "")
                elif t == "quota_exhausted":
                    exhausted = True
                    break
                elif t == "done":
                    break
            except Exception:
                pass
        return preview[:80].replace("\n", " "), exhausted
    except requests.exceptions.Timeout:
        return "TIMEOUT", False
    except Exception as e:
        return f"ERROR: {e}", False


print("=" * 68)
print("  Railway RAG  --  Key Exhaustion Stress Test")
print(f"  Target: {BASE}/ask/smart  |  {len(QUESTIONS)} unique questions")
print("=" * 68)

initial = check_key_status()
print(f"\n  Start: key[{initial.get('active_key_index', 0)}]"
      f"  fails={initial.get('quota_fail_count', 0)}/40\n")

for i, question in enumerate(QUESTIONS, 1):
    print(f"[Q{i:02d}/{len(QUESTIONS)}] {question[:60]}...")
    preview, exhausted = ask_question(question)

    if i % 5 == 0 or exhausted:
        st = check_key_status()
        key_idx  = st.get("active_key_index", "?")
        fail_cnt = st.get("quota_fail_count", 0)
        is_exh   = st.get("exhausted", False)
        bkdn     = st.get("key_fail_counts", {})
        bar      = "#" * (fail_cnt // 2)
        flag     = "[EXHAUSTED]" if is_exh else "[OK]"
        print(f"         >>> key[{key_idx}]  {fail_cnt:2d}/40 [{bar:<20}]  {flag}")
        if bkdn:
            print(f"             Per-key: {bkdn}")

    if exhausted:
        print("\n  [!!] QUOTA EXHAUSTED - key rotation triggered!")
        print("       Check the System Operational panel in the UI.")
        break

    time.sleep(PAUSE_SEC)

print()
print("=" * 68)
final = check_key_status()
print("FINAL STATE")
print(f"  Active Key     : Key #{final.get('active_key_index', '?')}")
print(f"  Quota Failures : {final.get('quota_fail_count', 0)} / 40")
print(f"  Exhausted      : {final.get('exhausted', False)}")
print(f"  Hours Left     : {final.get('hours_remaining_in_cooldown', 0)}")
print(f"  Key breakdown  : {final.get('key_fail_counts', {})}")
print("=" * 68)
