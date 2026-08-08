"""Quick unit test for the updated trim/direction logic in retriever.py."""
import sys, re
sys.stdout.reconfigure(encoding='utf-8')

# ── simulate the station terms ─────────────────────────────────────────────
all_station_terms = {"narasaraopet", "nrt", "bhimavaram", "bvrt"}
from_canon, from_code = "narasaraopet", "nrt"
to_canon,   to_code   = "bhimavaram",   "bvrt"

# ── NEW FORMAT doc (after re-ingestion) ───────────────────────────────────
new_doc = (
    "Train 18519 — Mumbai LTT Express (Daily). From VSKP to LTT. 24 stops, 1495.0 km.\n"
    "VSKP dep 23:20 | DVD arr 23:50 dep 23:52 (2min) | PAP arr 01:09 dep 01:10 (1min) | "
    "SLO arr 01:18 dep 01:20 (2min) | RJY arr 02:08 dep 02:10 (2min) | TNKU arr 03:08 dep 03:10 (2min) | "
    "BVRT arr 03:44 dep 03:45 (1min) | AKVD arr 04:04 dep 04:05 (1min) | KKLR arr 04:24 dep 04:25 (1min) | "
    "GDV arr 04:58 dep 05:00 (2min) | NRT arr 06:30 dep 06:45 (15min) | KZJ arr 10:28 dep 10:30 (2min) | "
    "LTT arr 22:30 [last]."
)

# ── OLD FORMAT doc ─────────────────────────────────────────────────────────
old_doc = (
    "Train 77248 (Daily). From BVRT to GNT. "
    "Stops (6): BVRT > NRT > GNT > OGL > TPTY > MAS. Distance: 320 km."
)

print("=" * 70)
print("TEST 1 — Trim NEW format doc")
print("=" * 70)
content = new_doc
parts    = content.split("\n", 1)
header   = parts[0]
sched    = parts[1] if len(parts) > 1 else ""
segments = [seg.strip() for seg in sched.rstrip(".").split("|") if seg.strip()]

print(f"Total segments: {len(segments)}")
trimmed_segs = []
for idx, seg in enumerate(segments):
    is_first  = (idx == 0)
    is_last   = (idx == len(segments) - 1)
    seg_code  = seg.split()[0].lower() if seg.split() else ""
    is_target = any(term in seg.lower() for term in all_station_terms) or seg_code in all_station_terms
    if is_first or is_last or is_target:
        trimmed_segs.append(seg)
        marker = "<-- KEPT (first)" if is_first else ("<-- KEPT (last)" if is_last else "<-- KEPT (target)")
        print(f"  [{idx:02d}] {seg[:55]:<55} {marker}")
    else:
        pass

trimmed = header + "\n" + " | ".join(trimmed_segs) + "."
print(f"\nResult ({len(new_doc)} chars → {len(trimmed)} chars):")
print(trimmed[:300])

print()
print("=" * 70)
print("TEST 2 — Direction check NEW format (BVRT before NRT = wrong direction for NRT→BVRT)")
print("=" * 70)

sched_line = new_doc.split("\n", 1)[1]
stops = [seg.strip().split()[0].lower() for seg in sched_line.split("|") if seg.strip() and seg.strip().split()]
print(f"Extracted codes: {stops[:8]}...")
fi = next((i for i, s in enumerate(stops) if from_canon in s or from_code in s), -1)
ti = next((i for i, s in enumerate(stops) if to_canon   in s or to_code   in s), -1)
print(f"from (NRT) at index: {fi}")
print(f"to   (BVRT) at index: {ti}")
print(f"Direction valid (NRT before BVRT): {fi < ti}")  # NRT is after BVRT in this train = WRONG direction

print()
print("=" * 70)
print("TEST 3 — Trim OLD format doc")
print("=" * 70)
content = old_doc
stops_idx    = content.index("Stops")
header       = content[:stops_idx]
stops_section = content[stops_idx:]
colon_idx    = stops_section.index(":")
stops_raw    = stops_section[colon_idx + 1:].strip()
all_stops    = [s.strip() for s in stops_raw.split(">") if s.strip()]
print(f"All stops: {all_stops}")
trimmed = [s for i,s in enumerate(all_stops)
           if i==0 or i==len(all_stops)-1 or any(t in s.lower() for t in all_station_terms)]
print(f"Trimmed:   {trimmed}")
print("✅ Tests complete")
