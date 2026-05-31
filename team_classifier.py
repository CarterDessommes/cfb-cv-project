"""
Step 4: Team Classifier using SigLIP + UMAP + K-Means.

Pipeline:
  1. Crop each player bounding box from the frame
  2. Embed each crop with SigLIP (vision encoder)
  3. Reduce embeddings to 2D with UMAP
  4. K-Means (k=2) to split into two teams

Usage:
    classifier = TeamClassifier()
    classifier.fit(frame, boxes)            # call once on a clean frame with all players visible
    labels = classifier.classify(frame, boxes)  # returns list of "offense" / "defense"
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel
from umap import UMAP
from sklearn.cluster import KMeans


_MODEL_ID = "google/siglip-base-patch16-224"
_MIN_CROP_PX = 10   # discard crops smaller than this in either dimension


def _best_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class TeamClassifier:
    """
    Classifies players into two teams using SigLIP embeddings + UMAP + K-Means.

    Call fit() once on a frame where most players are visible, then classify()
    on every subsequent frame.
    """

    def __init__(self, device: str | None = None):
        self.device = device or _best_device()
        print(f"Loading SigLIP on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self.model = SiglipVisionModel.from_pretrained(_MODEL_ID).to(self.device)
        self.model.eval()

        self._umap: UMAP | None = None
        self._kmeans: KMeans | None = None
        self._centroids: np.ndarray | None = None  # centroids in raw embedding space

        # 0 or 1 — which K-Means cluster is currently labeled "offense"
        self._offense_cluster: int = 0
        self._ball_votes: list[int] = []
        self._VOTE_WIN = 30

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crops(self, frame: np.ndarray, boxes: list) -> tuple[list, list[int]]:
        """Return (crops, valid_box_indices) — skips boxes that are too small."""
        crops, indices = [], []
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
            crop = frame[y1:y2, x1:x2]
            if crop.shape[0] >= _MIN_CROP_PX and crop.shape[1] >= _MIN_CROP_PX:
                crops.append(crop)
                indices.append(i)
        return crops, indices

    @torch.no_grad()
    def _embed(self, crops: list) -> np.ndarray:
        """Run SigLIP vision encoder on a list of BGR crops. Returns (N, D) array."""
        pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().float().numpy()  # (N, 768)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frame: np.ndarray, boxes: list) -> bool:
        """
        Fit the classifier on one frame.
        boxes: list of [x1, y1, x2, y2, ...] — extra fields are ignored.
        Returns True on success, False if too few players were found.
        """
        crops, _ = self._crops(frame, boxes)
        if len(crops) < 4:
            print(f"fit: only {len(crops)} valid crops, need at least 4 — skipping")
            return False

        print(f"fit: embedding {len(crops)} player crops...")
        embeddings = self._embed(crops)  # (N, 768)

        # UMAP: reduce to 2D for clustering
        n_neighbors = min(15, len(crops) - 1)
        self._umap = UMAP(n_components=2, n_neighbors=n_neighbors, random_state=42)
        reduced = self._umap.fit_transform(embeddings)  # (N, 2)

        # K-Means on 2D UMAP space
        self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        self._kmeans.fit(reduced)

        # Also store per-cluster mean in raw embedding space for fast classify()
        labels = self._kmeans.labels_
        self._centroids = np.stack([
            embeddings[labels == 0].mean(axis=0),
            embeddings[labels == 1].mean(axis=0),
        ])

        print("fit: done — centroids locked")
        return True

    def classify(self, frame: np.ndarray, boxes: list) -> list[str]:
        """
        Classify each box as 'team_a' or 'team_b'.
        Uses nearest centroid in raw SigLIP embedding space (no UMAP needed per frame).
        Returns 'unknown' for boxes with invalid crops.
        """
        if self._centroids is None:
            raise RuntimeError("Call fit() before classify().")

        crops, valid_idx = self._crops(frame, boxes)
        out = ["unknown"] * len(boxes)

        if not crops:
            return out

        embeddings = self._embed(crops)  # (N, 768)
        for i, (emb, box_idx) in enumerate(zip(embeddings, valid_idx)):
            d0 = np.linalg.norm(emb - self._centroids[0])
            d1 = np.linalg.norm(emb - self._centroids[1])
            cluster = 0 if d0 <= d1 else 1
            out[box_idx] = "offense" if cluster == self._offense_cluster else "defense"

        return out

    def update_offense_from_ball(self, ball_xy, boxes: list, labels: list[str]) -> None:
        """
        Call after classify() each frame. Votes over a rolling window on which
        cluster is actually offense (closer to the ball). Flips the assignment
        once a majority is reached so the classifier self-corrects.
        ball_xy: (cx, cy) in pixel coords, or None if ball not detected.
        """
        if ball_xy is None or self._centroids is None:
            return

        bx, by = float(ball_xy[0]), float(ball_xy[1])
        off_pts, def_pts = [], []
        for box, lbl in zip(boxes, labels):
            if lbl == "unknown":
                continue
            cx = (float(box[0]) + float(box[2])) / 2
            cy = (float(box[1]) + float(box[3])) / 2
            (off_pts if lbl == "offense" else def_pts).append((cx, cy))

        if not off_pts or not def_pts:
            return

        off_c = np.mean(off_pts, axis=0)
        def_c = np.mean(def_pts, axis=0)
        d_off = np.hypot(bx - off_c[0], by - off_c[1])
        d_def = np.hypot(bx - def_c[0], by - def_c[1])

        # vote 1 = current "offense" cluster is actually farther from ball (needs swap)
        self._ball_votes.append(1 if d_def < d_off else 0)
        if len(self._ball_votes) > self._VOTE_WIN:
            self._ball_votes.pop(0)

        if sum(self._ball_votes) > len(self._ball_votes) // 2:
            self._offense_cluster ^= 1
            self._ball_votes.clear()

# ------------------------------------------------------------------
# Quick visual test:  python team_classifier.py <video> [--model PATH]
# ------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from ultralytics import YOLO

    if len(sys.argv) < 2:
        print("Usage: python team_classifier.py <video_path> [--model PATH]")
        sys.exit(1)

    video_path = sys.argv[1]
    model_path = "weights/player-best.pt"
    for i, arg in enumerate(sys.argv[2:], 2):
        if arg == "--model" and i + 1 < len(sys.argv):
            model_path = sys.argv[i + 1]

    detector = YOLO(model_path)
    classifier = TeamClassifier()
    fitted = False

    COLORS = {"offense": (0, 200, 255), "defense": (255, 100, 0), "unknown": (128, 128, 128)}

    cap = cv2.VideoCapture(video_path)
    frame_num = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_num += 1

        results = detector.track(frame, persist=True, conf=0.4, verbose=False,
                                  device=_best_device(), half=True)

        boxes = []
        if results[0].boxes is not None and results[0].boxes.id is not None:
            xyxy = results[0].boxes.xyxy.cpu().numpy()
            ids  = results[0].boxes.id.cpu().numpy().astype(int)
            clss = results[0].boxes.cls.cpu().numpy().astype(int)
            for box, tid, cls in zip(xyxy, ids, clss):
                if cls == 0:   # players only
                    boxes.append([*box, tid, cls])

        if boxes and not fitted:
            fitted = classifier.fit(frame, boxes)

        if fitted and boxes:
            team_labels = classifier.classify(frame, boxes)
            for box, label in zip(boxes, team_labels):
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                tid = int(box[4])
                color = COLORS[label]
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{label.upper()} #{tid}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        cv2.putText(frame, f"Frame {frame_num}  offense=blue  defense=orange",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Team Classifier", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
