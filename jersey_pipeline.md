# Jersey Number Prediction & Maintenance in `pipeline.py`

An exhaustive, step-by-step walkthrough of how jersey numbers flow through
[pipeline.py](pipeline.py) — from raw frame to the stable, persistent number
drawn on each player.

---

## The big picture

Jersey-number handling is a **5-stage funnel** applied every frame:

1. **Crop** — cut a region likely to contain the chest number ([pipeline.py:125-146](pipeline.py#L125-L146))
2. **Gate (orientation)** — drop side-facing players whose number can't be read ([pipeline.py:348-356](pipeline.py#L348-L356))
3. **Read (OCR)** — classify each crop into a number, with a blur gate and confidence gate ([pipeline.py:221-249](pipeline.py#L221-L249))
4. **Stabilize** — fold the per-frame guess into a per-tracklet temporal vote, and eventually *lock* it ([pipeline.py:30-43](pipeline.py#L30-L43))
5. **Render** — draw the stable number and feed it into the field state ([pipeline.py:366-382](pipeline.py#L366-L382))

The key design idea: a single frame's read is noisy and untrustworthy, so the
system **never commits to a single-frame guess**. It accumulates
confidence-weighted votes per tracking ID over a sliding window and locks in once
enough agreement is seen.

Two module-level dictionaries carry this state across frames:

```python
_NUMBER_HISTORY: dict[int, list[tuple[str, float]]] = {}   # track_id -> recent (pred, conf) reads
_NUMBER_LOCKED:  dict[int, str] = {}                        # track_id -> permanently decided number
```

Both are **cleared at the start of every `run_pipeline` call**
([pipeline.py:282-283](pipeline.py#L282-L283)) so state doesn't leak between videos.

---

## Stage 0 — Prerequisites for the frame

Before any jersey logic runs, each frame goes through detection/tracking and pose:

- **Detection + tracking** ([pipeline.py:316-328](pipeline.py#L316-L328)):
  `detector.track(..., persist=True)` runs a YOLO tracker. Only class `0`
  (players) is kept. Each entry in `boxes` becomes
  `[x1, y1, x2, y2, track_id, cls]`. The `track_id` is the linchpin of the whole
  stabilization system — it's the key under which a player's number history is
  accumulated.
- **Pose estimation** ([pipeline.py:341-344](pipeline.py#L341-L344)): runs only
  every `pose_every` frames (default 3) for speed, caching the result in
  `cached_pose_kps`. `get_keypoints_for_boxes`
  ([pipeline.py:257-276](pipeline.py#L257-L276)) runs the pose model on the whole
  frame, then **matches** each pose skeleton to a tracked player box by best IoU
  (must exceed 0.3, else `None`). So each player box gets either a 17-keypoint
  array or `None`.

Pose is used **only for gating decisions** (is this player big/forward enough to
bother reading?), never for the crop actually fed to the reader — that's an
explicit design note at [pipeline.py:117-120](pipeline.py#L117-L120).

---

## Stage 1 — Cropping ([pipeline.py:125-146](pipeline.py#L125-L146))

`_crops()` iterates every player box and calls `_number_crop()`:

```python
_CROP_BOTTOM    = 0.95   # keep from box top down to 95% of player height
_CROP_WIDTH_PAD = 0.15   # widen 15% past bbox left/right
```

`_number_crop` deliberately takes a **generous** crop — nearly the whole player,
padded 15% wider than the bounding box and extending down to 95% of the player's
height. The rationale ([pipeline.py:126-127](pipeline.py#L126-L127)): the chest
number should *never* be clipped by keypoint error, an off-center number, or
out-flung arms. It's a wide net.

Note there's a **second, narrower crop function**, `_pose_crop`
([pipeline.py:81-104](pipeline.py#L81-L104)), which tightly frames the torso using
shoulder/hip keypoints. **It is not used in the live path** — `_crops` calls
`_number_crop`, not `_pose_crop`. `_pose_crop` exists for dataset creation
symmetry (the comment block explains the model is trained on
`_number_crop`-style inputs so inference matches).

Each surviving crop must be at least `_MIN_CROP_PX = 10` px on both sides
([pipeline.py:141](pipeline.py#L141)). `_crops` returns three parallel lists:
`crops`, `valid_idx` (the index back into `boxes`), and `rects`.

---

## Stage 2 — Orientation gate ([pipeline.py:348-356](pipeline.py#L348-L356))

For each candidate crop, the code looks up that player's pose keypoints and calls
`_is_forward_facing()` ([pipeline.py:107-114](pipeline.py#L107-L114)):

```python
shoulder_span = abs(rs[0] - ls[0])          # horizontal distance between shoulders
bbox_w = max(box width, 1.0)
return (shoulder_span / bbox_w) > 0.25      # _SIDE_FACING_THRESHOLD
```

The intuition: when a player faces the camera, their shoulders span a wide
horizontal distance relative to their bounding box. When they're **side-on**, the
shoulders collapse toward each other (small span), and the chest number is
invisible or distorted. If the ratio falls below 0.25, the crop is **dropped** —
no point reading an unreadable number.

Safety fallbacks: if either shoulder keypoint has confidence below
`_KP_CONF_MIN = 0.3`, or if there are no keypoints at all, the gate **passes by
default** ([pipeline.py:110-111](pipeline.py#L110-L111),
[pipeline.py:352-353](pipeline.py#L352-L353)) — it won't filter out a player just
because pose failed.

The output is `gated_crops` plus `gated_box_idx` (which box each surviving crop
maps to).

---

## Stage 3 — OCR reading ([pipeline.py:226-249](pipeline.py#L226-L249))

`jersey_reader.predict(gated_crops)` runs all gated crops through the YOLO
**classification** model in one batched call ([pipeline.py:236](pipeline.py#L236)).
For each crop, two gates apply:

**a) Blur gate** ([pipeline.py:239-242](pipeline.py#L239-L242)):
```python
blur_var = cv2.Laplacian(crop, cv2.CV_64F).var()
if blur_var < _BLUR_THRESHOLD:   # 20
    out.append((None, [], blur_var))   # rejected: no number this frame
    continue
```
Laplacian variance measures edge sharpness. A motion-blurred or tiny crop has low
variance → it's discarded before its (unreliable) prediction can pollute the vote.

**b) Confidence gate** ([pipeline.py:243-248](pipeline.py#L243-L248)):
```python
top5      = [(name, conf), ...]              # the model's 5 best guesses
top1_label, top1_conf = top5[0]
top1 = (top1_label, top1_conf) if top1_conf >= _OCR_CONF_THRESHOLD else None  # 0.35
```
The classifier's single best guess is only accepted as a real read (`top1`) if
its confidence ≥ 0.35. Otherwise `top1` is `None`. The full `top5` and `blur_var`
are still returned — these feed the debug panel regardless.

So `predict` returns one `(top1, top5, blur_var)` tuple per crop. Back in the loop
([pipeline.py:360-363](pipeline.py#L360-L363)), only crops with a non-`None`
`top1` populate `ocr_number_map`, keyed by box index → `(label, conf)`.

---

## Stage 4 — Temporal stabilization ([pipeline.py:30-43](pipeline.py#L30-L43)) — the heart of "maintained"

This is where a flickering per-frame guess becomes a stable number. For every
player, `_stable_number(track_id, pred, conf)` is called
([pipeline.py:369](pipeline.py#L369)):

```python
def _stable_number(track_id, pred, conf):
    if track_id in _NUMBER_LOCKED:
        return _NUMBER_LOCKED[track_id]        # (1) locked → never changes again
    history = _NUMBER_HISTORY.setdefault(track_id, [])
    history.append((pred, conf))               # (2) record this frame's read
    if len(history) > _VOTE_WINDOW:            # _VOTE_WINDOW = 75 (~2.5s @ 30fps)
        history.pop(0)                          # (3) sliding window — drop oldest
    weights = {}
    for p, c in history:
        weights[p] = weights.get(p, 0.0) + c   # (4) sum confidence per candidate number
    best = max(weights, key=weights.__getitem__)  # (5) confidence-weighted winner
    if sum(1 for p, _ in history if p == best) >= _LOCK_VOTES:  # _LOCK_VOTES = 8
        _NUMBER_LOCKED[track_id] = best         # (6) lock permanently
    return best
```

Walking through the mechanics:

1. **Lock short-circuit**: once a tracklet is locked, the function returns the
   locked value immediately and ignores all further reads. The number is frozen
   for the life of that track ID.
2. **Accumulate**: each accepted read (only reads that passed blur + confidence +
   orientation gates ever reach here) is appended to that track's history as
   `(predicted_number, confidence)`.
3. **Sliding window**: history is capped at 75 entries (~2.5 s). Older reads age
   out, so a number can drift if a player is consistently re-read differently —
   *until* it locks.
4. **Confidence-weighted voting**: rather than a plain count, each candidate
   number's votes are weighted by the model's confidence. A few high-confidence
   reads can outweigh many low-confidence ones.
5. **Winner**: the number with the highest summed confidence is the current
   answer.
6. **Locking**: if the winning number appears in **at least 8 frames** within the
   window, it's written to `_NUMBER_LOCKED` and becomes permanent.

Players with **no accepted read this frame** (`raw is None`) get `"?"` and are
*not* added to history ([pipeline.py:368-369](pipeline.py#L368-L369)) — missing
frames don't dilute the vote.

**Important caveat / subtlety:** the lock criterion is a raw *count* of frames
matching `best` (`>= 8`), but `best` is chosen by *confidence weight*. These can
disagree in edge cases — e.g., a number with high cumulative confidence but only
7 frames won't lock yet, while the count is what gates the lock. Also note the
count uses the current window, so the 8 matching frames must coexist in the
75-frame window.

---

## Stage 5 — Rendering & field state ([pipeline.py:365-382](pipeline.py#L365-L382))

For each player box:
```python
track_id = int(box[4])
raw = ocr_number_map.get(i)                          # this frame's accepted read, or None
number = _stable_number(track_id, *raw) if raw else "?"
```
The stable number goes into the per-player `field_state` dict (alongside
`track_id`, `bbox`, `team`) and is drawn on the frame with the team color. An
**"L" suffix** is appended to the on-screen label when the track is locked
([pipeline.py:379](pipeline.py#L379),
[pipeline.py:381-382](pipeline.py#L381-L382)) — e.g. `OFF #42L`.

The number is **not** propagated into the downstream field-mapping step —
`detections_for_mapper` ([pipeline.py:385-388](pipeline.py#L385-L388)) only
carries `track_id`, `bbox`, and `class`. So jersey numbers live in `field_state`
and the rendered overlay, but the top-down projection doesn't currently use them.

---

## Stage 6 — The debug panel ([pipeline.py:169-218](pipeline.py#L169-L218))

A right-hand side panel (`_build_ocr_panel`) is built from `gated_crops` +
`ocr_results`. For each gated player it shows: the track ID, the crop's blur
variance, a thumbnail of the actual crop, and the **top-5** predictions with
confidences (winner highlighted green, or "BLURRY (filtered)" if the blur gate
killed it). This is purely diagnostic — it's `hstack`'d next to the frame
([pipeline.py:404](pipeline.py#L404)) so you can watch the reader's reasoning live.

---

## Summary of the tunable constants

| Constant | Value | Role |
|---|---|---|
| `_MIN_CROP_PX` | 10 | Minimum crop dimension to attempt OCR |
| `_SIDE_FACING_THRESHOLD` | 0.25 | shoulder-span/bbox ratio below which a player is "side-on" and skipped |
| `_KP_CONF_MIN` | 0.3 | Min keypoint confidence to trust the orientation gate |
| `_BLUR_THRESHOLD` | 20 | Min Laplacian variance; below = too blurry to read |
| `_OCR_CONF_THRESHOLD` | 0.35 | Min classifier confidence to accept a read |
| `_VOTE_WINDOW` | 75 | Sliding-window length (~2.5 s @ 30 fps) for voting |
| `_LOCK_VOTES` | 8 | Frames agreeing on a number before it locks permanently |
| `pose_every` | 3 | Run pose every N frames (gating only) |

---

The throughline: **detection gives stable track IDs → cropping +
orientation/blur/confidence gates filter out unreadable inputs → surviving reads
are accumulated per track ID into a confidence-weighted sliding-window vote → the
vote produces the displayed number and, after 8 agreeing frames, locks it
permanently.** That layering is what turns a jittery frame-by-frame classifier
into a number that stays put on each player.
