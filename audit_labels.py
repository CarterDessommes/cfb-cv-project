"""
Audit a labels.tsv produced by label_crops.py.

Buckets crops by jersey number, reports the per-number sample count, and flags
numbers with too few samples to train on reliably. Use this before fine-tuning
to find under-represented numbers (linemen, bench players) and to gauge how
much of the clip was wasted on "unreadable" crops.

Usage:
    python3 audit_labels.py <out-dir>/labels.tsv [--min N]

    --min N   numbers with fewer than N readable crops are flagged (default 10)
"""

import sys
import csv
from collections import Counter


def audit(labels_path: str, min_samples: int) -> None:
    counts: Counter[str] = Counter()
    total = 0
    with open(labels_path, newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            total += 1
            counts[row["label"]] += 1

    unreadable = counts.pop("unreadable", 0)
    readable   = sum(counts.values())

    print(f"Total crops:       {total}")
    print(f"Readable:          {readable}")
    print(f"Unreadable:        {unreadable} "
          f"({unreadable / max(total, 1) * 100:.0f}%)")
    print(f"Distinct numbers:  {len(counts)}\n")

    if not counts:
        print("No readable numbers found.")
        return

    # Sort by jersey number when numeric, so the table reads naturally
    def _key(item):
        label, _ = item
        return (0, int(label)) if label.isdigit() else (1, label)

    print(f"{'number':>8}  {'count':>6}  bar")
    print("-" * 40)
    flagged = []
    for label, n in sorted(counts.items(), key=_key):
        bar  = "#" * min(n, 30)
        warn = "  <-- LOW" if n < min_samples else ""
        if n < min_samples:
            flagged.append((label, n))
        print(f"{label:>8}  {n:>6}  {bar}{warn}")

    print()
    if flagged:
        print(f"{len(flagged)} number(s) below --min {min_samples}: "
              + ", ".join(f"{lbl}({n})" for lbl, n in flagged))
        print("Collect more crops for these (lower --every, longer clip) "
              "before training.")
    else:
        print(f"All numbers have >= {min_samples} samples.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 audit_labels.py <labels.tsv> [--min N]")
        sys.exit(1)

    labels_path = sys.argv[1]
    min_samples = 10
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--min" and i + 1 < len(sys.argv):
            min_samples = int(sys.argv[i + 1])

    audit(labels_path, min_samples)
