"""
Step 4: Team Classifier using SigLIP + K-Means.

Pipeline:
  1. Crop each player bounding box from the frame
  2. Embed each crop with SigLIP (vision encoder)
  3. K-Means (k=2) directly on the raw embeddings to split into two teams

(K-Means on raw SigLIP embeddings produces the same split as the previous
UMAP→K-Means reduction on the benchmark videos, without umap-learn's
multi-second first-call JIT cost.)

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
    Classifies players into two teams using SigLIP embeddings + K-Means.

    Call fit() once on a frame where most players are visible, then classify()
    on every subsequent frame.
    """

    def __init__(self, device: str | None = None):
        self.device = device or _best_device()
        print(f"Loading SigLIP on {self.device}...")
        self.processor = AutoImageProcessor.from_pretrained(_MODEL_ID)
        self.model = SiglipVisionModel.from_pretrained(_MODEL_ID).to(self.device)
        self.model.eval()

        self._kmeans = None
        self._centroids: np.ndarray | None = None  # centroids in raw embedding space
        self._track_clusters: dict[int, int] = {}

        # 0 or 1 — which K-Means cluster is labeled "offense"; set at fit() time
        self._offense_cluster: int = 0

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

    @staticmethod
    def _track_id(box) -> int | None:
        """Return the tracker id from a detection box, if one is present."""
        if len(box) < 5:
            return None
        try:
            return int(box[4])
        except (TypeError, ValueError):
            return None

    def warmup(self) -> None:
        """Build the MPS/CUDA graphs once so the first real embed isn't slow."""
        self._embed([np.zeros((64, 48, 3), dtype=np.uint8)])

    @torch.no_grad()
    def _embed(self, crops: list) -> np.ndarray:
        """Run SigLIP vision encoder on a list of BGR crops. Returns (N, D) array."""
        pil = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        return outputs.pooler_output.cpu().float().numpy()  # (N, 768)

    def _cluster_for_embedding(self, emb: np.ndarray) -> int:
        d0 = np.linalg.norm(emb - self._centroids[0])
        d1 = np.linalg.norm(emb - self._centroids[1])
        return 0 if d0 <= d1 else 1

    def _label_for_cluster(self, cluster: int) -> str:
        return "offense" if cluster == self._offense_cluster else "defense"

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, frame: np.ndarray, boxes: list, ball_xy=None) -> bool:
        """
        Fit the classifier on one frame.
        boxes: list of [x1, y1, x2, y2, ...] — extra fields are ignored.
        ball_xy: (cx, cy) pixel position of the ball, used to label which
                 cluster is offense (the cluster containing the player nearest
                 the ball). If None, cluster 0 is arbitrarily labeled offense.
        Returns True on success, False if too few players were found.
        """
        crops, valid_idx = self._crops(frame, boxes)
        if len(crops) < 4:
            print(f"fit: only {len(crops)} valid crops, need at least 4 — skipping")
            return False

        from sklearn.cluster import KMeans

        print(f"fit: embedding {len(crops)} player crops...")
        embeddings = self._embed(crops)  # (N, 768)

        # K-Means directly on the raw embedding space (see module docstring).
        self._kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        self._kmeans.fit(embeddings)

        # Store per-cluster mean in raw embedding space for fast classify()
        labels = self._kmeans.labels_
        self._centroids = np.stack([
            embeddings[labels == 0].mean(axis=0),
            embeddings[labels == 1].mean(axis=0),
        ])
        self._track_clusters.clear()
        for box_idx, cluster in zip(valid_idx, labels):
            track_id = self._track_id(boxes[box_idx])
            if track_id is not None:
                self._track_clusters[track_id] = int(cluster)

        # Label offense as the cluster containing the player nearest the ball.
        # The center is physically on the ball at the snap, so this is a clean
        # one-shot read. Falls back to cluster 0 when ball is not yet detected.
        if ball_xy is not None:
            bx, by = float(ball_xy[0]), float(ball_xy[1])
            nearest_cluster = 0
            nearest_dist    = float("inf")
            for box_idx, cluster in zip(valid_idx, labels):
                box = boxes[box_idx]
                cx  = (float(box[0]) + float(box[2])) / 2
                cy  = (float(box[1]) + float(box[3])) / 2
                dist = np.hypot(bx - cx, by - cy)
                if dist < nearest_dist:
                    nearest_dist    = dist
                    nearest_cluster = int(cluster)
            self._offense_cluster = nearest_cluster
            print(f"fit: offense = cluster {nearest_cluster} "
                  f"(nearest player {nearest_dist:.0f} px from ball)")
        else:
            print("fit: ball not detected — offense label is arbitrary (cluster 0)")

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

        uncached_crops = []
        uncached_box_idx = []
        for crop, box_idx in zip(crops, valid_idx):
            track_id = self._track_id(boxes[box_idx])
            if track_id is not None and track_id in self._track_clusters:
                out[box_idx] = self._label_for_cluster(self._track_clusters[track_id])
            else:
                uncached_crops.append(crop)
                uncached_box_idx.append(box_idx)

        if uncached_crops:
            embeddings = self._embed(uncached_crops)  # (N, 768)
            for emb, box_idx in zip(embeddings, uncached_box_idx):
                cluster = self._cluster_for_embedding(emb)
                track_id = self._track_id(boxes[box_idx])
                if track_id is not None:
                    self._track_clusters[track_id] = cluster
                out[box_idx] = self._label_for_cluster(cluster)

        return out

    def assign_clusters(self, frame: np.ndarray, boxes: list) -> list[int]:
        """
        Return the nearest-centroid cluster index (0 or 1) per box.
        Returns -1 for boxes whose crops are invalid/too small.

        Unlike classify(), this returns the raw cluster — not an offense/defense
        label — so callers can cache it per track_id and re-derive the label each
        frame from the fixed _offense_cluster set at fit() time.
        """
        return [c for c, _ in self.assign_clusters_margins(frame, boxes)]

    def assign_clusters_margins(self, frame: np.ndarray, boxes: list) -> list[tuple[int, float]]:
        """
        Like assign_clusters, but returns (cluster, margin) per box, where
        margin is the relative gap between the distances to the two centroids
        (0 = equidistant/ambiguous crop, larger = more decisive). Invalid crops
        yield (-1, 0.0).
        """
        if self._centroids is None:
            raise RuntimeError("Call fit() before assign_clusters().")

        crops, valid_idx = self._crops(frame, boxes)
        out = [(-1, 0.0)] * len(boxes)
        if not crops:
            return out

        embeddings = self._embed(crops)  # (N, 768)
        for emb, box_idx in zip(embeddings, valid_idx):
            d0 = float(np.linalg.norm(emb - self._centroids[0]))
            d1 = float(np.linalg.norm(emb - self._centroids[1]))
            mean = (d0 + d1) / 2.0
            margin = abs(d0 - d1) / mean if mean > 0 else 0.0
            out[box_idx] = (0 if d0 <= d1 else 1, margin)
        return out

    def cluster_to_label(self, cluster: int) -> str:
        """Map a cluster index (0/1, or -1 for unknown) to its team label."""
        if cluster < 0:
            return "unknown"
        return self._label_for_cluster(cluster)

    def forget(self, track_id: int) -> None:
        """Drop a cached track id if the tracker recycles it."""
        self._track_clusters.pop(int(track_id), None)

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
