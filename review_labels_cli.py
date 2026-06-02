#!/usr/bin/env python3
"""CLI label reviewer for game_db/labels.tsv

Usage:
  - Enter → keep current label, advance
  - Type + Enter → replace label, advance
  - b → go back one image
  - q → save and quit
"""

from pathlib import Path
import csv
import sys
import subprocess

LABELS_TSV = Path(__file__).parent / "game_db" / "labels.tsv"


def load_labels(path: Path):
    with path.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
        fieldnames = reader.fieldnames
    return rows, fieldnames


def save_labels(path: Path, rows, fieldnames):
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def main():
    if not LABELS_TSV.exists():
        print(f"labels file not found: {LABELS_TSV}")
        sys.exit(1)

    rows, fieldnames = load_labels(LABELS_TSV)
    if not rows:
        print("No rows in labels.tsv")
        return

    idx = 0
    total = len(rows)

    try:
        while True:
            row = rows[idx]
            filename = row.get("filename", "")
            current = row.get("label", "")
            print(f"[{idx+1}/{total}] {filename}\n  current label: {current}")
            # Open the image in the default macOS viewer so the user can see it
            img_path = Path(__file__).parent / "game_db" / "crops" / filename
            if img_path.exists():
                try:
                    # Open in macOS without bringing Preview to foreground
                    # -g = do not bring the application to the foreground
                    subprocess.Popen(["open", "-g", str(img_path)])
                except Exception:
                    try:
                        # Fallback: AppleScript 'without activating'
                        subprocess.Popen([
                            "osascript", "-e",
                            f'tell application "Preview" to open POSIX file "{img_path}" without activating'
                        ])
                    except Exception:
                        # best-effort; continue even if open fails
                        pass
            resp = input("New label (Enter=keep, b=back, q=quit): ").strip()
            if resp == "":
                if idx < total - 1:
                    idx += 1
                    continue
                else:
                    print("End of list.")
                    break
            if resp.lower() == "q":
                break
            if resp.lower() == "b":
                if idx > 0:
                    idx -= 1
                continue
            # set new label and save
            row["label"] = resp
            save_labels(LABELS_TSV, rows, fieldnames)
            if idx < total - 1:
                idx += 1
            else:
                print("End of list.")
                break
    except (KeyboardInterrupt, EOFError):
        print("\nInterrupted — saving and exiting.")

    save_labels(LABELS_TSV, rows, fieldnames)
    print(f"Saved {LABELS_TSV}")


if __name__ == "__main__":
    main()
