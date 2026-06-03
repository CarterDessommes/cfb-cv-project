"""
annotate.py — local keypoint annotation tool.

Usage:
    python annotate.py                          # uses bijon_frames/ and annotations.json
    python annotate.py --frames path/to/frames  # custom frames directory
    python annotate.py --ann path/to/ann.json   # custom annotations file

Open http://localhost:5050 in your browser.
Click a keypoint name on the left, then click the image to place it.
Click an existing dot to remove it. Changes save automatically.
"""

import argparse
import base64
import json
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from field_schema import KEYPOINTS, FIELD_LANDMARKS

_FIELD_WIDTH  = 53.33
_FIELD_LENGTH = 100.0

app = Flask(__name__)

# -- Config (set by main) --
FRAMES_DIR: Path = Path("bijon_frames")
ANN_PATH:   Path = Path("annotations.json")
_annotations: dict = {}
_frame_names: list = []


def _load_annotations():
    global _annotations
    if ANN_PATH.exists():
        _annotations = json.loads(ANN_PATH.read_text())
    else:
        _annotations = {}


def _save_annotations():
    ANN_PATH.write_text(json.dumps(_annotations, indent=2))


def _frame_key(fname: str) -> str:
    return fname  # e.g. "frame_00000.jpg"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return HTML_PAGE


@app.route("/frames")
def list_frames():
    return jsonify(_frame_names)


@app.route("/frame/<fname>")
def get_frame(fname):
    return send_from_directory(str(FRAMES_DIR), fname)


@app.route("/annotations", methods=["GET"])
def get_annotations():
    return jsonify(_annotations)


@app.route("/annotations/<fname>", methods=["POST"])
def save_frame_annotations(fname):
    kpts = request.json  # {str(kp_id): [x, y]} or {} to clear
    key = _frame_key(fname)
    if kpts:
        _annotations[key] = {int(k): v for k, v in kpts.items()}
    elif key in _annotations:
        del _annotations[key]
    _save_annotations()
    return jsonify({"ok": True})


@app.route("/keypoints")
def get_keypoints():
    return jsonify(KEYPOINTS)


@app.route("/overlay/<fname>")
def get_overlay(fname):
    """Return projected field lines as pixel-space polylines for the current frame."""
    ann = _annotations.get(_frame_key(fname), {})
    if len(ann) < 4:
        return jsonify({"lines": []})

    kp_dict = {int(k): tuple(v) for k, v in ann.items()}
    img_pts   = np.array([kp_dict[k]        for k in sorted(kp_dict)], dtype=np.float32)
    field_pts = np.array([FIELD_LANDMARKS[k] for k in sorted(kp_dict)], dtype=np.float32)
    H, _ = cv2.findHomography(img_pts, field_pts, cv2.RANSAC, 5.0)
    if H is None:
        return jsonify({"lines": []})
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        return jsonify({"lines": []})

    def to_px(fx, fy):
        pt = cv2.perspectiveTransform(np.array([[[fx, fy]]], dtype=np.float64), H_inv)[0][0]
        return [round(float(pt[0]), 1), round(float(pt[1]), 1)]

    lines = []
    # Major yard lines (every 10 yds) — draw through all 4 rail y positions
    for x in range(0, 101, 10):
        lines.append({"type": "yard10",
                      "pts": [to_px(x, y) for y in [0.0, 23.58, 29.75, 53.33]]})
    # Minor yard lines (every 5 yds) — sideline to sideline only
    for x in range(5, 100, 10):
        lines.append({"type": "yard5",
                      "pts": [to_px(x, 0.0), to_px(x, 53.33)]})
    # Sidelines
    for y in [0.0, _FIELD_WIDTH]:
        lines.append({"type": "sideline",
                      "pts": [to_px(x, y) for x in range(0, 101, 5)]})
    # Hash lines
    for y in [23.58, 29.75]:
        lines.append({"type": "hash",
                      "pts": [to_px(x, y) for x in range(0, 101, 5)]})

    return jsonify({"lines": lines})


# ---------------------------------------------------------------------------
# HTML + JS (single-page app)
# ---------------------------------------------------------------------------

HTML_PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Field Annotator</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body { display: flex; height: 100vh; font-family: monospace; background: #1a1a1a; color: #eee; }

#sidebar {
  width: 220px; min-width: 220px; background: #111;
  display: flex; flex-direction: column; border-right: 1px solid #333;
}
#nav {
  padding: 8px; display: flex; gap: 6px; align-items: center;
  border-bottom: 1px solid #333; flex-wrap: wrap;
}
#nav button {
  padding: 4px 10px; background: #333; color: #eee;
  border: 1px solid #555; border-radius: 3px; cursor: pointer; font-size: 12px;
}
#nav button:hover { background: #444; }
#frame-info { font-size: 11px; color: #aaa; padding: 4px 8px; border-bottom: 1px solid #222; }
#kp-list { flex: 1; overflow-y: auto; padding: 4px; }
.kp-btn {
  display: block; width: 100%; text-align: left;
  padding: 4px 6px; margin: 1px 0; font-size: 11px;
  background: #1e1e1e; border: 1px solid #333; border-radius: 3px;
  color: #ccc; cursor: pointer; white-space: nowrap; overflow: hidden;
}
.kp-btn:hover { background: #2a2a2a; }
.kp-btn.selected { background: #1a4a1a; border-color: #4caf50; color: #8f8; }
.kp-btn.placed { border-color: #f90; color: #fa0; }
.kp-btn.selected.placed { background: #2a3a00; border-color: #cf0; color: #ef0; }

#main { flex: 1; display: flex; flex-direction: column; align-items: center;
  justify-content: center; overflow: hidden; padding: 8px; }
#canvas-wrap { position: relative; display: inline-block; }
#img { max-width: 100%; max-height: calc(100vh - 60px); display: block; }
#overlay { position: absolute; top: 0; left: 0; width: 100%; height: 100%; cursor: crosshair; }
#status { font-size: 11px; color: #aaa; padding: 4px 0; }
</style>
</head>
<body>
<div id="sidebar">
  <div id="nav">
    <button onclick="prevFrame()">◀ Prev</button>
    <button onclick="nextFrame()">Next ▶</button>
    <button onclick="clearFrame()" style="color:#f88">Clear</button>
    <button id="overlay-btn" onclick="toggleOverlay()" style="color:#0ff">Grid ✓</button>
  </div>
  <div id="frame-info">—</div>
  <div id="kp-list"></div>
</div>
<div id="main">
  <div id="status">Loading...</div>
  <div id="canvas-wrap">
    <img id="img" src="">
    <canvas id="overlay"></canvas>
  </div>
</div>

<script>
let frames = [], frameIdx = 0;
let keypoints = [];
let selectedKp = null;
let placed = {};   // {kp_id: [x_frac, y_frac]}  fractions of image size
let annotations = {};  // all loaded annotations

const img     = document.getElementById('img');
const overlay = document.getElementById('overlay');
const ctx     = overlay.getContext('2d');
const status  = document.getElementById('status');
const frameInfo = document.getElementById('frame-info');

async function init() {
  [frames, keypoints, annotations] = await Promise.all([
    fetch('/frames').then(r => r.json()),
    fetch('/keypoints').then(r => r.json()),
    fetch('/annotations').then(r => r.json()),
  ]);
  buildKpList();
  loadFrame(0);
}

function buildKpList() {
  const list = document.getElementById('kp-list');
  list.innerHTML = '';
  keypoints.forEach(kp => {
    const btn = document.createElement('button');
    btn.className = 'kp-btn';
    btn.id = `kp-${kp.id}`;
    btn.textContent = `[${kp.id}] ${kp.name}`;
    btn.onclick = () => selectKp(kp.id);
    list.appendChild(btn);
  });
}

function selectKp(id) {
  selectedKp = id;
  document.querySelectorAll('.kp-btn').forEach(b => b.classList.remove('selected'));
  const btn = document.getElementById(`kp-${id}`);
  if (btn) btn.classList.add('selected');
  status.textContent = `Placing: [${id}] ${keypoints.find(k=>k.id===id)?.name}  — click image`;
}

function loadFrame(idx) {
  frameIdx = idx;
  const fname = frames[idx];
  img.src = `/frame/${fname}`;
  frameInfo.textContent = `${idx + 1}/${frames.length}  ${fname}`;

  placed = {};

  function _loadPlaced() {
    const ann = annotations[fname] || {};
    placed = {};
    const pw = img.naturalWidth, ph = img.naturalHeight;
    Object.entries(ann).forEach(([kid, xy]) => {
      // Stored as pixel coords; convert to fractions for the overlay
      placed[parseInt(kid)] = pw > 0 ? [xy[0] / pw, xy[1] / ph] : [0, 0];
    });
  }

  img.onload = () => { _loadPlaced(); syncOverlay(); redraw(); updateKpButtons(); };
  if (img.complete) { _loadPlaced(); syncOverlay(); redraw(); updateKpButtons(); }
}

function syncOverlay() {
  overlay.width  = img.offsetWidth;
  overlay.height = img.offsetHeight;
}

function redraw() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  Object.entries(placed).forEach(([kid, [xf, yf]]) => {
    const kp = keypoints.find(k => k.id === parseInt(kid));
    const cx = xf * overlay.width;
    const cy = yf * overlay.height;
    const isSel = selectedKp === parseInt(kid);
    ctx.beginPath();
    ctx.arc(cx, cy, isSel ? 7 : 5, 0, Math.PI * 2);
    ctx.fillStyle = isSel ? '#ef0' : '#f90';
    ctx.fill();
    ctx.strokeStyle = '#000';
    ctx.lineWidth = 1;
    ctx.stroke();
    ctx.fillStyle = '#fff';
    ctx.font = '10px monospace';
    ctx.fillText(kp ? kp.id : kid, cx + 7, cy - 4);
  });
}

function updateKpButtons() {
  document.querySelectorAll('.kp-btn').forEach(b => b.classList.remove('placed'));
  Object.keys(placed).forEach(kid => {
    const btn = document.getElementById(`kp-${kid}`);
    if (btn) btn.classList.add('placed');
  });
}

let _dragKp   = null;   // kp_id being dragged, or null
let _dragMoved = false; // did the mouse move enough to count as a drag?

overlay.addEventListener('mousedown', e => {
  const rect = overlay.getBoundingClientRect();
  const xf = (e.clientX - rect.left) / overlay.width;
  const yf = (e.clientY - rect.top)  / overlay.height;
  // Hit-test all placed dots
  for (const [kid, [ex, ey]] of Object.entries(placed)) {
    const dx = (ex - xf) * overlay.width;
    const dy = (ey - yf) * overlay.height;
    if (Math.hypot(dx, dy) < 14) {
      _dragKp    = parseInt(kid);
      _dragMoved = false;
      e.preventDefault();
      return;
    }
  }
  _dragKp = null;
  _dragMoved = false;
});

overlay.addEventListener('mousemove', e => {
  const rect = overlay.getBoundingClientRect();
  const xf = (e.clientX - rect.left) / overlay.width;
  const yf = (e.clientY - rect.top)  / overlay.height;

  if (_dragKp !== null) {
    _dragMoved    = true;
    placed[_dragKp] = [xf, yf];
    overlay.style.cursor = 'grabbing';
    redraw();
    return;
  }

  // Cursor hint when hovering over any placed dot
  let onDot = false;
  for (const [, [ex, ey]] of Object.entries(placed)) {
    if (Math.hypot((ex - xf) * overlay.width, (ey - yf) * overlay.height) < 14) {
      onDot = true; break;
    }
  }
  overlay.style.cursor = onDot ? 'grab' : 'crosshair';
});

overlay.addEventListener('mouseup', e => {
  overlay.style.cursor = 'crosshair';

  if (_dragKp !== null && _dragMoved) {
    // Finished a drag — save the new position
    saveFrame();
    _dragKp = null;
    return;
  }

  const wasClean = (_dragKp === null) || !_dragMoved;
  _dragKp = null;
  if (!wasClean) return;

  // ── Original click logic: place or delete the selected keypoint ──
  if (selectedKp === null) { status.textContent = 'Select a keypoint first.'; return; }
  const rect = overlay.getBoundingClientRect();
  const xf = (e.clientX - rect.left) / overlay.width;
  const yf = (e.clientY - rect.top)  / overlay.height;

  if (placed[selectedKp]) {
    const [ex, ey] = placed[selectedKp];
    const dx = (ex - xf) * overlay.width;
    const dy = (ey - yf) * overlay.height;
    if (Math.hypot(dx, dy) < 12) {
      delete placed[selectedKp];
      saveFrame();
      redraw(); updateKpButtons(); return;
    }
  }

  placed[selectedKp] = [xf, yf];
  saveFrame();
  redraw(); updateKpButtons();
});

async function saveFrame() {
  const fname = frames[frameIdx];
  // Convert fractions back to pixel coords using natural image size
  const pw = img.naturalWidth, ph = img.naturalHeight;
  const out = {};
  Object.entries(placed).forEach(([kid, [xf, yf]]) => {
    out[kid] = [xf * pw, yf * ph];
  });
  await fetch(`/annotations/${fname}`, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(out),
  });
  // Update local cache
  annotations[fname] = out;
  status.textContent = `Saved ${Object.keys(placed).length} keypoints for ${fname}`;
}

async function clearFrame() {
  placed = {};
  await fetch(`/annotations/${frames[frameIdx]}`, {
    method: 'POST', headers: {'Content-Type': 'application/json'}, body: '{}',
  });
  delete annotations[frames[frameIdx]];
  redraw(); updateKpButtons();
  status.textContent = 'Cleared.';
}

function prevFrame() { if (frameIdx > 0) loadFrame(frameIdx - 1); }
function nextFrame() { if (frameIdx < frames.length - 1) loadFrame(frameIdx + 1); }

document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft')  prevFrame();
  if (e.key === 'ArrowRight') nextFrame();
  if (e.key === 'ArrowUp') {
    e.preventDefault();
    const idx = keypoints.findIndex(k => k.id === selectedKp);
    const next = idx <= 0 ? keypoints.length - 1 : idx - 1;
    selectKp(keypoints[next].id);
    document.getElementById(`kp-${keypoints[next].id}`)?.scrollIntoView({block:'nearest'});
  }
  if (e.key === 'ArrowDown') {
    e.preventDefault();
    const idx = keypoints.findIndex(k => k.id === selectedKp);
    const next = idx < 0 || idx === keypoints.length - 1 ? 0 : idx + 1;
    selectKp(keypoints[next].id);
    document.getElementById(`kp-${keypoints[next].id}`)?.scrollIntoView({block:'nearest'});
  }
});

window.addEventListener('resize', () => { syncOverlay(); redraw(); });

// ── Field grid overlay ─────────────────────────────────────────────────────
let showOverlay = true;
let overlayLines = [];

function toggleOverlay() {
  showOverlay = !showOverlay;
  document.getElementById('overlay-btn').textContent = showOverlay ? 'Grid ✓' : 'Grid ✗';
  redraw();
}

async function fetchOverlay() {
  if (!frames.length) return;
  const fname = frames[frameIdx];
  const res = await fetch(`/overlay/${fname}`);
  const data = await res.json();
  overlayLines = data.lines || [];
  redraw();
}

function drawOverlay() {
  if (!showOverlay || !overlayLines.length) return;
  const scaleX = overlay.width  / img.naturalWidth;
  const scaleY = overlay.height / img.naturalHeight;

  const colors = {
    yard10:   'rgba(0,255,255,0.85)',
    yard5:    'rgba(0,200,200,0.45)',
    sideline: 'rgba(255,160,0,0.9)',
    hash:     'rgba(180,180,255,0.7)',
  };
  const widths = { yard10: 1.5, yard5: 1, sideline: 2, hash: 1 };

  for (const line of overlayLines) {
    const pts = line.pts.map(([x, y]) => [x * scaleX, y * scaleY]);
    ctx.beginPath();
    ctx.strokeStyle = colors[line.type] || 'rgba(255,255,255,0.6)';
    ctx.lineWidth   = widths[line.type] || 1;
    ctx.moveTo(pts[0][0], pts[0][1]);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][0], pts[i][1]);
    ctx.stroke();
  }
}

// Patch redraw to include overlay
const _redrawBase = redraw;
redraw = function() {
  _redrawBase();
  drawOverlay();
};

// Patch saveFrame to refresh overlay after each click
const _saveFrameBase = saveFrame;
saveFrame = async function() {
  await _saveFrameBase();
  fetchOverlay();
};

// Patch loadFrame to fetch overlay for new frame
const _loadFrameBase = loadFrame;
loadFrame = function(idx) {
  overlayLines = [];
  _loadFrameBase(idx);
  fetchOverlay();
};

init();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global FRAMES_DIR, ANN_PATH

    parser = argparse.ArgumentParser(description="Local keypoint annotation tool")
    parser.add_argument("--frames", default="bijon_frames", help="Directory of frame images")
    parser.add_argument("--ann",    default="annotations.json", help="Annotations JSON file")
    parser.add_argument("--port",   type=int, default=5050)
    args = parser.parse_args()

    FRAMES_DIR = Path(args.frames)
    ANN_PATH   = Path(args.ann)

    global _frame_names
    _frame_names = sorted(p.name for p in FRAMES_DIR.glob("*.jpg"))
    if not _frame_names:
        print(f"No .jpg frames found in {FRAMES_DIR}")
        return

    _load_annotations()
    print(f"Loaded {len(_frame_names)} frames from {FRAMES_DIR}")
    print(f"Annotations file: {ANN_PATH}")
    print(f"Open http://localhost:{args.port} in your browser")

    app.run(port=args.port, debug=False)


if __name__ == "__main__":
    main()
