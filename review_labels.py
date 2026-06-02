#!/usr/bin/env python3
"""
Label Reviewer — GUI for game_db crops.

Keyboard shortcuts
  Enter / →      advance (keep or apply typed label)
  ← / BackSpace  go back one image (only when text entry is empty)
  u              mark "unreadable" and advance
  0-9            append digit to label entry
  Ctrl+Z         undo last label change
  Escape / q     save and quit
"""

import tkinter as tk
from tkinter import ttk, font as tkfont
from pathlib import Path
import pandas as pd
from PIL import Image, ImageTk

CROPS_DIR    = Path(__file__).parent / "game_db" / "crops"
LABELS_TSV   = Path(__file__).parent / "game_db" / "labels.tsv"
DISPLAY_SIZE = 640   # image panel size (px)

# quick-label buttons shown on the right panel
QUICK_LABELS = ["unreadable", "0", "1", "5", "7", "8", "55", "63", "65", "70", "71", "89"]

BG          = "#1e1e1e"
BG2         = "#252526"
FG          = "#d4d4d4"
FG_DIM      = "#6a6a6a"
ACCENT      = "#4ec9b0"
ACCENT2     = "#569cd6"
CHANGED     = "#dcdcaa"
BTN_BG      = "#2d2d2d"
BTN_ACTIVE  = "#3a3a3a"
BORDER      = "#3a3a3a"


class Reviewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Label Reviewer")
        self.root.configure(bg=BG)
        self.root.resizable(True, True)

        self.df = pd.read_csv(LABELS_TSV, sep="\t", dtype={"label": str})
        self._orig_labels = self.df["label"].copy()  # for undo
        self._undo_stack: list[tuple[int, str]] = []  # (idx, old_label)

        self.idx = 0
        self._filter_label = tk.StringVar(value="All")
        self._filtered_indices: list[int] = list(range(len(self.df)))

        self._build_ui()
        self._apply_filter()

    # ── UI construction ──────────────────────────────────────────────────────

    def _build_ui(self):
        mono   = tkfont.Font(family="Menlo", size=11)
        mono_s = tkfont.Font(family="Menlo", size=10)
        big    = tkfont.Font(family="Menlo", size=20, weight="bold")
        big_s  = tkfont.Font(family="Menlo", size=14, weight="bold")

        # ── top bar ──────────────────────────────────────────────────────────
        top = tk.Frame(self.root, bg=BG2, pady=6)
        top.pack(fill="x", padx=0, pady=0)

        self._progress_var = tk.StringVar()
        tk.Label(top, textvariable=self._progress_var, font=mono,
                 bg=BG2, fg=FG_DIM).pack(side="left", padx=12)

        self._pbar_var = tk.DoubleVar()
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("dark.Horizontal.TProgressbar",
                        troughcolor=BG, background=ACCENT,
                        bordercolor=BG2, lightcolor=ACCENT, darkcolor=ACCENT)
        pbar = ttk.Progressbar(top, variable=self._pbar_var, maximum=100,
                               length=200, style="dark.Horizontal.TProgressbar")
        pbar.pack(side="left", padx=8)

        self._changed_var = tk.StringVar()
        tk.Label(top, textvariable=self._changed_var, font=mono_s,
                 bg=BG2, fg=CHANGED).pack(side="left", padx=8)

        # Filter
        tk.Label(top, text="Filter:", font=mono_s, bg=BG2, fg=FG_DIM).pack(
            side="left", padx=(20, 4))
        unique_labels = sorted(self.df["label"].unique().tolist())
        filter_choices = ["All"] + unique_labels
        self._filter_menu = ttk.Combobox(top, textvariable=self._filter_label,
                                         values=filter_choices, width=12,
                                         font=mono_s, state="readonly")
        self._filter_menu.pack(side="left")
        self._filter_menu.bind("<<ComboboxSelected>>", self._on_filter_change)

        # Jump-to
        tk.Label(top, text="  Jump:", font=mono_s, bg=BG2, fg=FG_DIM).pack(
            side="left", padx=(12, 4))
        self._jump_var = tk.StringVar()
        jump_entry = tk.Entry(top, textvariable=self._jump_var, font=mono_s,
                              width=6, bg=BTN_BG, fg=FG, insertbackground=FG,
                              relief="flat", highlightthickness=1,
                              highlightcolor=ACCENT, highlightbackground=BORDER)
        jump_entry.pack(side="left")
        jump_entry.bind("<Return>", self._jump_to)
        tk.Button(top, text="Go", font=mono_s, bg=BTN_BG, fg=FG,
                  activebackground=BTN_ACTIVE, activeforeground=FG,
                  relief="flat", padx=6, command=self._jump_to).pack(
            side="left", padx=4)

        # ── main body: image  |  right panel ─────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(12, 0))

        # left: image + filename
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        self._filename_var = tk.StringVar()
        tk.Label(left, textvariable=self._filename_var, font=mono_s,
                 bg=BG, fg=FG_DIM).pack(anchor="w")

        self._img_label = tk.Label(left, bg="#111", width=DISPLAY_SIZE,
                                   height=DISPLAY_SIZE)
        self._img_label.pack(pady=(4, 0))

        # right: current label, quick buttons, entry, nav
        right = tk.Frame(body, bg=BG, padx=16)
        right.pack(side="right", fill="y")

        tk.Label(right, text="Current label", font=mono_s,
                 bg=BG, fg=FG_DIM).pack(anchor="w", pady=(0, 2))
        self._current_var = tk.StringVar()
        self._current_lbl = tk.Label(right, textvariable=self._current_var,
                                     font=big, bg=BG, fg=ACCENT, width=12,
                                     anchor="w")
        self._current_lbl.pack(anchor="w")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(right, text="Quick labels", font=mono_s,
                 bg=BG, fg=FG_DIM).pack(anchor="w", pady=(0, 4))

        # unreadable gets its own wide button
        self._make_btn(right, "unreadable", wide=True)

        # number buttons in rows of 4
        nums = [l for l in QUICK_LABELS if l != "unreadable"]
        row_frame = None
        for i, lbl in enumerate(nums):
            if i % 4 == 0:
                row_frame = tk.Frame(right, bg=BG)
                row_frame.pack(anchor="w", pady=2)
            self._make_btn(row_frame, lbl, wide=False)

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        tk.Label(right, text="Custom label  (Enter = apply)",
                 font=mono_s, bg=BG, fg=FG_DIM).pack(anchor="w", pady=(0, 4))
        self._entry_var = tk.StringVar()
        self._entry = tk.Entry(right, textvariable=self._entry_var, font=big_s,
                               width=10, bg=BTN_BG, fg=FG,
                               insertbackground=FG, relief="flat",
                               highlightthickness=1, highlightcolor=ACCENT,
                               highlightbackground=BORDER)
        self._entry.pack(anchor="w")
        self._entry.focus_set()

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        nav = tk.Frame(right, bg=BG)
        nav.pack(anchor="w")
        tk.Button(nav, text="← Back", font=mono_s, bg=BTN_BG, fg=FG,
                  activebackground=BTN_ACTIVE, activeforeground=FG,
                  relief="flat", padx=10, pady=4,
                  command=self._go_back).pack(side="left", padx=(0, 8))
        tk.Button(nav, text="Keep →", font=mono_s, bg=BTN_BG, fg=ACCENT,
                  activebackground=BTN_ACTIVE, activeforeground=ACCENT,
                  relief="flat", padx=10, pady=4,
                  command=self._advance).pack(side="left")

        ttk.Separator(right, orient="horizontal").pack(fill="x", pady=10)

        # stats
        tk.Label(right, text="Label counts", font=mono_s,
                 bg=BG, fg=FG_DIM).pack(anchor="w", pady=(0, 4))
        self._stats_var = tk.StringVar()
        tk.Label(right, textvariable=self._stats_var, font=mono_s,
                 bg=BG, fg=FG, justify="left", anchor="w").pack(anchor="w")

        # ── bottom status bar ─────────────────────────────────────────────────
        status = tk.Frame(self.root, bg=BG2, pady=4)
        status.pack(fill="x", side="bottom")
        tk.Label(status,
                 text="u=unreadable  0-9=digit  ←/→=navigate  Ctrl+Z=undo  q=quit",
                 font=mono_s, bg=BG2, fg=FG_DIM).pack(side="left", padx=12)
        self._save_var = tk.StringVar(value="")
        tk.Label(status, textvariable=self._save_var, font=mono_s,
                 bg=BG2, fg=ACCENT2).pack(side="right", padx=12)

        # ── key bindings ──────────────────────────────────────────────────────
        self.root.bind("<Return>",    self._advance)
        self.root.bind("<Right>",     self._advance)
        self.root.bind("<Left>",      self._go_back)
        self.root.bind("<BackSpace>", self._go_back)
        self.root.bind("<Escape>",    self._quit)
        self.root.bind("q",           self._quit)
        self.root.bind("u",           self._mark_unreadable)
        self.root.bind("<Control-z>", self._undo)
        # digit shortcuts → fill entry (only when entry not focused)
        for d in "0123456789":
            self.root.bind(d, self._digit_shortcut)
        self.root.protocol("WM_DELETE_WINDOW", self._quit)

    def _make_btn(self, parent, label_text: str, wide: bool):
        w = 18 if wide else 4
        btn = tk.Button(
            parent, text=label_text, font=tkfont.Font(family="Menlo", size=11),
            bg=BTN_BG, fg=FG, activebackground=BTN_ACTIVE, activeforeground=ACCENT,
            relief="flat", padx=6, pady=3, width=w,
            command=lambda lv=label_text: self._quick_label(lv),
        )
        btn.pack(side="left" if not wide else "top",
                 anchor="w", pady=2, padx=(0, 4) if not wide else 0)

    # ── filter / jump ─────────────────────────────────────────────────────────

    def _apply_filter(self):
        fv = self._filter_label.get()
        if fv == "All":
            self._filtered_indices = list(range(len(self.df)))
        else:
            self._filtered_indices = [
                i for i, lbl in enumerate(self.df["label"]) if str(lbl) == fv
            ]
        self.idx = 0
        if self._filtered_indices:
            self._show()
        else:
            self._filename_var.set("(no images match filter)")
            self._current_var.set("")
            self._img_label.configure(image="", text="No matches")

    def _on_filter_change(self, _event=None):
        self._apply_filter()
        self._entry.focus_set()

    def _jump_to(self, _event=None):
        raw = self._jump_var.get().strip()
        if not raw:
            return
        try:
            n = int(raw)
        except ValueError:
            return
        # n is 1-based position within the filtered list
        n = max(1, min(n, len(self._filtered_indices)))
        self.idx = n - 1
        self._show()
        self._jump_var.set("")
        self._entry.focus_set()

    # ── display ───────────────────────────────────────────────────────────────

    def _show(self):
        if not self._filtered_indices:
            return
        df_idx = self._filtered_indices[self.idx]
        row    = self.df.iloc[df_idx]
        pos    = self.idx + 1
        total  = len(self._filtered_indices)

        self._progress_var.set(f"{pos} / {total}")
        self._pbar_var.set(100 * pos / total)
        self._filename_var.set(row["filename"])

        label = str(row["label"])
        self._current_var.set(label)
        # colour changed labels differently
        orig = str(self._orig_labels.iloc[df_idx])
        self._current_lbl.configure(fg=CHANGED if label != orig else ACCENT)

        self._entry_var.set("")

        # changed-count badge
        n_changed = (self.df["label"] != self._orig_labels).sum()
        self._changed_var.set(f"{n_changed} changed" if n_changed else "")

        # image
        img_path = CROPS_DIR / row["filename"]
        if img_path.exists():
            img = Image.open(img_path)
            scale = DISPLAY_SIZE / max(img.width, img.height)
            new_w = max(1, int(img.width  * scale))
            new_h = max(1, int(img.height * scale))
            img = img.resize((new_w, new_h), Image.LANCZOS)
            self._photo = ImageTk.PhotoImage(img)
            self._img_label.configure(image=self._photo, text="",
                                      width=DISPLAY_SIZE, height=DISPLAY_SIZE)
        else:
            self._img_label.configure(image="", text="[image not found]",
                                      width=DISPLAY_SIZE, height=DISPLAY_SIZE)

        # stats (top-10 by count)
        counts = self.df["label"].value_counts().head(10)
        lines = [f"  {lbl:<12} {cnt}" for lbl, cnt in counts.items()]
        self._stats_var.set("\n".join(lines))

    # ── actions ───────────────────────────────────────────────────────────────

    def _commit_label(self, new_val: str):
        """Apply new_val to the current row and push to undo stack."""
        df_idx = self._filtered_indices[self.idx]
        old    = str(self.df.at[df_idx, "label"])
        if new_val != old:
            self._undo_stack.append((df_idx, old))
            self.df.at[df_idx, "label"] = new_val
            self._save()

    def _advance(self, _event=None):
        new_val = self._entry_var.get().strip()
        if new_val:
            self._commit_label(new_val)
        if self.idx < len(self._filtered_indices) - 1:
            self.idx += 1
            self._show()
        else:
            self._quit()

    def _go_back(self, _event=None):
        if self._entry_var.get() == "" and self.idx > 0:
            self.idx -= 1
            self._show()

    def _quick_label(self, label_text: str):
        self._commit_label(label_text)
        if self.idx < len(self._filtered_indices) - 1:
            self.idx += 1
            self._show()
        else:
            self._quit()

    def _mark_unreadable(self, _event=None):
        # ignore if user is typing in a text field
        focused = self.root.focus_get()
        if isinstance(focused, tk.Entry):
            return
        self._quick_label("unreadable")

    def _digit_shortcut(self, event):
        focused = self.root.focus_get()
        if isinstance(focused, tk.Entry):
            return  # let the entry handle it normally
        self._entry_var.set(self._entry_var.get() + event.char)
        self._entry.focus_set()
        self._entry.icursor("end")

    def _undo(self, _event=None):
        if not self._undo_stack:
            return
        df_idx, old_label = self._undo_stack.pop()
        self.df.at[df_idx, "label"] = old_label
        self._save()
        # jump to that image in the filtered list
        if df_idx in self._filtered_indices:
            self.idx = self._filtered_indices.index(df_idx)
        self._show()

    # ── persistence ───────────────────────────────────────────────────────────

    def _save(self):
        self.df.to_csv(LABELS_TSV, sep="\t", index=False)
        self._save_var.set("saved")
        self.root.after(1500, lambda: self._save_var.set(""))

    def _quit(self, _event=None):
        self._save()
        self.root.destroy()


def main():
    root = tk.Tk()
    root.configure(bg=BG)
    root.minsize(820, 560)
    root.geometry("+100+80")   # place near top-left so it's always on screen
    Reviewer(root)
    root.lift()
    root.attributes("-topmost", True)
    root.after(200, lambda: root.attributes("-topmost", False))  # un-pin after raise
    root.mainloop()


if __name__ == "__main__":
    main()
