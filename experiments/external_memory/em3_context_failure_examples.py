from __future__ import annotations
import json
from collections import Counter, defaultdict, deque
from pathlib import Path

H = 5000
AUTHORS = {"Etinjat", "Re_spectators", "breaddddd"}
N_EACH = 5
ROOT = Path(r"C:\Users\chiar\Desktop\LBH")
PILOT = ROOT / r"thesis-personalisation\results\personalisation\pilot_a_context_memory"
FOUR = ROOT / r"thesis-context-lab\results\personalisation\external_memory\em2_four_way_dev_compare\rows.jsonl"
SURFACE = ROOT / r"thesis-context-lab\results\personalisation\external_memory\em2_fixed_gfc_dev\selected_rows.jsonl"
HISTORY = PILOT / "history_manifest.jsonl"
DEV = PILOT / "dev_manifest.jsonl"

def rows(path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def get(r, *names, default=None):
    for n in names:
        if n in r:
            return r[n]
    return default

def rid(r): return str(get(r, "row_id", "condition_id", "id"))
def author(r): return str(get(r, "author", "user_id"))
def pos(r, fallback):
    x = get(r, "chronological_position", "position", "interaction_position", "query_position", default=None)
    return fallback if x is None else int(x)
def pinyin(r):
    x = get(r, "pinyin_segments", "segmented_pinyin", "pinyin")
    if isinstance(x, (list, tuple)): return tuple(str(v) for v in x)
    return (str(x),)
def gold(r): return str(get(r, "gold", "target", "target_candidate", "current_gold"))
def context(r): return str(get(r, "context", "preceding_context", "current_context", default="") or "")

def rank_candidates(cands, lf, lc):
    return sorted(cands, key=lambda c: (-(float(c["normalized_generic_score"]) + lf*float(c["frequency_support"]) + lc*float(c["context_support"])), int(c["generic_rank"])))

four = {rid(r): r for r in rows(FOUR)}
surface = {rid(r): r for r in rows(SURFACE)}
dev_rows = [r for r in rows(DEV) if author(r) in AUTHORS and rid(r) in four]

all_by_author = defaultdict(list)
idx = 0
for r in rows(HISTORY):
    if author(r) in AUTHORS:
        all_by_author[author(r)].append((pos(r, idx), idx, False, r))
    idx += 1
for r in dev_rows:
    all_by_author[author(r)].append((pos(r, idx), idx, True, r))
    idx += 1

cases = {"G_RIGHT_HIDDEN_WRONG": [], "F_RIGHT_HIDDEN_WRONG": []}

for a in sorted(AUTHORS):
    data = sorted(all_by_author[a], key=lambda x: (x[0], x[1]))
    hist = deque(maxlen=H)
    i = 0
    while i < len(data):
        cur_pos = data[i][0]
        j = i
        group = []
        while j < len(data) and data[j][0] == cur_pos:
            group.append(data[j]); j += 1

        for qpos, _, is_dev, q in group:
            qid = rid(q)
            if not is_dev or qid not in four:
                continue
            ff = four[qid]
            # A missing/None rank means the Gold was not ranked at Top-1.
            # Do not cast None with int().
            def is_top1(value):
                if value is None:
                    return False
                try:
                    return int(value) == 1
                except (TypeError, ValueError):
                    return False

            g_ok = is_top1(ff.get("G_rank"))
            f_ok = is_top1(ff.get("F_rank"))
            h_ok = is_top1(ff.get("Hidden_M1_rank"))
            if not ((g_ok and not h_ok) or (f_ok and not h_ok)):
                continue

            qpy, qgold = pinyin(q), gold(q)
            same = [h for _, h in hist if pinyin(h) == qpy]
            positives = [h for h in same if gold(h) == qgold]
            negatives = [h for h in same if gold(h) != qgold]
            counts = Counter(gold(h) for h in same)

            g_pred = f_pred = h_pred = "(prediction text unavailable)"
            srow = surface.get(qid)
            if srow and "ranking" in srow:
                cands = srow["ranking"]
                g_pred = rank_candidates(cands,0,0)[0]["candidate"]
                f_pred = rank_candidates(cands,4,0)[0]["candidate"]
                h_pred = rank_candidates(cands,0,4)[0]["candidate"]

            rec = {
                "row_id": qid, "author": a, "pinyin": " ".join(qpy), "gold": qgold,
                "G": g_pred, "F": f_pred, "Hidden_M1": h_pred,
                "positive_count": len(positives), "negative_count": len(negatives),
                "distinct_targets": len(counts),
                "gold_share": len(positives)/len(same) if same else 0.0,
                "target_counts": counts.most_common(5),
                "context": context(q)[-140:].replace("\n"," "),
                "positive_examples": [{"target": gold(h), "context": context(h)[-110:].replace("\n"," ")} for h in positives[-3:]],
                "negative_examples": [{"target": gold(h), "context": context(h)[-110:].replace("\n"," ")} for h in negatives[-2:]],
            }
            if g_ok and not h_ok: cases["G_RIGHT_HIDDEN_WRONG"].append(rec)
            if f_ok and not h_ok: cases["F_RIGHT_HIDDEN_WRONG"].append(rec)

        for p0, _, _, r in group: hist.append((p0, r))
        i = j

def print_case(title, recs):
    print("\n" + "#"*110)
    print(title)
    print("TOTAL CASES:", len(recs))
    pc = sorted(r["positive_count"] for r in recs)
    if pc:
        mid = len(pc)//2
        med = pc[mid] if len(pc)%2 else (pc[mid-1]+pc[mid])/2
        print(f"POSITIVE COUNT: mean={sum(pc)/len(pc):.2f} median={med} min={min(pc)} max={max(pc)}")
    useful = [r for r in recs if r["positive_count"] >= 3]
    useful.sort(key=lambda r: (-r["positive_count"], -r["negative_count"], r["author"], r["row_id"]))
    for r in useful[:N_EACH]:
        print("\n" + "="*110)
        print("ROW:", r["row_id"])
        print("AUTHOR:", r["author"])
        print("PINYIN:", r["pinyin"])
        print("GOLD:", r["gold"])
        print("G:", r["G"], "| F:", r["F"], "| Hidden-M1:", r["Hidden_M1"])
        print("POSITIVE:", r["positive_count"], "| NEGATIVE:", r["negative_count"], "| DISTINCT TARGETS:", r["distinct_targets"], f"| GOLD SHARE: {r['gold_share']:.3f}")
        print("TOP TARGET COUNTS:", r["target_counts"])
        print("CURRENT CONTEXT:", r["context"])
        print("\nPOSITIVE HISTORY EXAMPLES:")
        for i,x in enumerate(r["positive_examples"],1): print(f"  P{i}: {x['target']} | {x['context']}")
        print("\nNEGATIVE HISTORY EXAMPLES:")
        for i,x in enumerate(r["negative_examples"],1): print(f"  N{i}: {x['target']} | {x['context']}")

print_case("A) GENERIC RIGHT / HIDDEN-M1 WRONG", cases["G_RIGHT_HIDDEN_WRONG"])
print_case("B) FREQUENCY RIGHT / HIDDEN-M1 WRONG", cases["F_RIGHT_HIDDEN_WRONG"])