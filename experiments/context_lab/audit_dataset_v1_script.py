import json
from collections import Counter, defaultdict
from pathlib import Path
from opencc import OpenCC

DATASET = Path(
    r"C:\Users\chiar\Desktop\LBH\thesis-deep-author\.build\dataset-v1-reconstruction"
    r"\data\processed\deep_author\interactions_t1_ready.jsonl"
)

cc = OpenCC("t2s")

total = 0
gold_changed = 0
context_changed = 0
both_changed = 0

by_author = defaultdict(lambda: Counter(
    total=0,
    gold_changed=0,
    context_changed=0,
    both_changed=0,
))

gold_examples = []
context_examples = []

with DATASET.open("r", encoding="utf-8") as f:
    for line in f:
        row = json.loads(line)

        author = str(row["author_name"])
        gold = str(row["gold"])
        context = str(row["context"])

        gold_t2s = cc.convert(gold)
        context_t2s = cc.convert(context)

        g = gold_t2s != gold
        c = context_t2s != context

        total += 1
        by_author[author]["total"] += 1

        if g:
            gold_changed += 1
            by_author[author]["gold_changed"] += 1
            if len(gold_examples) < 20:
                gold_examples.append((author, gold, gold_t2s))

        if c:
            context_changed += 1
            by_author[author]["context_changed"] += 1
            if len(context_examples) < 20:
                context_examples.append(
                    (author, context[-100:], context_t2s[-100:])
                )

        if g and c:
            both_changed += 1
            by_author[author]["both_changed"] += 1

        if total % 100000 == 0:
            print(f"rows scanned = {total:,}", flush=True)

print("\n=== DATASET V1 SCRIPT AUDIT ===")
print("rows =", total)
print(
    "gold_changed =",
    gold_changed,
    f"({gold_changed / total:.4%})"
)
print(
    "context_changed =",
    context_changed,
    f"({context_changed / total:.4%})"
)
print(
    "both_changed =",
    both_changed,
    f"({both_changed / total:.4%})"
)

print("\n=== BY AUTHOR ===")
for author in sorted(by_author):
    x = by_author[author]
    print(
        author,
        "rows=", x["total"],
        "gold_changed=", x["gold_changed"],
        f"({x['gold_changed']/x['total']:.4%})",
        "context_changed=", x["context_changed"],
        f"({x['context_changed']/x['total']:.4%})",
    )

print("\n=== GOLD EXAMPLES ===")
for x in gold_examples:
    print(repr(x))

print("\n=== CONTEXT EXAMPLES ===")
for x in context_examples:
    print(repr(x))
