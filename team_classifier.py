"""
Step 4: Team Classifier using SigLIP + K-Means.

Pipeline:
  1. Crop each player bounding box from the frame
  2. Embed each crop with SigLIP (vision encoder)
  3. K-Means (k=2) directly on the 768-D embeddings to split into two teams
  4. On subsequent frames, assign each player to the nearest centroid

Offense vs defense labeling:
  Ball proximity voting — the player physically closest to the ball is almost
  always offense (center at the snap, ball-carrier on a run, QB/receiver on a pass).
  A rolling majority vote over _VOTE_WIN frames flips the offense/defense assignment
  when the current labeling is consistently wrong.  A cooldown prevents oscillation.

Usage:
    classifier = TeamClassifier()
    classifier.fit(frame, boxes)
    labels = classifier.classify(frame, boxes)   # "offense" / "defense" / "unknown"
"""

import cv2
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, SiglipVisionModel
from sklearn.cluster import KMeans


_MODEL_ID    = "google/siglip-base-patch16-224"
_MIN_CROP_PX = 10


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
        self.model     = SiglipVisionModel.from_pretrained(_MODEL_ID).to(self.device)
        self.model.eval()

        self._centroids: np.ndarray | None = None  # (2, 768)

        self._offense_cluster: int  = 0   # which K-Means cluster is currently "offense"
        self._ball_votes: list[int] = []
        self._VOTE_WIN              = 30
        self._flip_cooldown: int    = 0   # frames remaining before another flip is allowed

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _crops(self, frame: np.ndarray, boxes: list) -> tuple[list, list[int]]:
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
        pil    = [Image.fromarray(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in crops]
        inputs = self.processor(images=pil, return_tensors="pt").to(self.device)
        return self.model(**inputs).pooler_output.cpu().float().numpy()  # (N, 768)

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
            print(f"fit: only {len(crops)} valid crops, need ≥4 — skipping")
            return False

        print(f"fit: embedding {len(crops)} player crops...")
        embeddings = self._embed(crops)                          # (N, 768)

        # K-Means directly in SigLIP embedding space — consistent with classify()
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        labels = kmeans.fit_predict(embeddings)

        self._centroids = np.stack([
            embeddings[labels == 0].mean(axis=0),
            embeddings[labels == 1].mean(axis=0),
        ])
        print("fit: done — centroids locked")
        return True

    def classify(self, frame: np.ndarray, boxes: list) -> list[str]:
        """
        Classify each box as 'offense', 'defense', or 'unknown'.
        Uses nearest centroid in raw SigLIP embedding space.
        """
        if self._centroids is None:
            raise RuntimeError("Call fit() before classify().")

        crops, valid_idx = self._crops(frame, boxes)
        out = ["unknown"] * len(boxes)
        if not crops:
            return out

        embeddings = self._embed(crops)
        for emb, box_idx in zip(embeddings, valid_idx):
            d0 = np.linalg.norm(emb - self._centroids[0])
            d1 = np.linalg.norm(emb - self._centroids[1])
            cluster = 0 if d0 <= d1 else 1
            out[box_idx] = "offense" if cluster == self._offense_cluster else "defense"

        return out

    def update_offense_from_ball(
        self, ball_xy, boxes: list, labels: list[str]
    ) -> bool:
        """
        Vote on which cluster is offense using ball proximity, then flip if needed.

        Strategy: the player *closest* to the ball is almost always offense
        (center at snap, ball-carrier on a run, QB/receiver on a pass).
        Returns True if a flip occurred (caller should re-classify this frame).
        """
        if ball_xy is None or self._centroids is None:
            return False

        if self._flip_cooldown > 0:
            self._flip_cooldown -= 1
            return False

        bx, by = float(ball_xy[0]), float(ball_xy[1])

        # Find the player closest to the ball
        best_dist  = float("inf")
        best_label = None
        for box, lbl in zip(boxes, labels):
            if lbl == "unknown":
                continue
            cx   = (float(box[0]) + float(box[2])) / 2
            cy   = (float(box[1]) + float(box[3])) / 2
            dist = np.hypot(bx - cx, by - cy)
            if dist < best_dist:
                best_dist  = dist
                best_label = lbl

        if best_label is None:
            return False

        # vote=1: nearest player is labeled "defense" → current assignment may be backwards
        self._ball_votes.append(0 if best_label == "offense" else 1)
        if len(self._ball_votes) > self._VOTE_WIN:
            self._ball_votes.pop(0)

        if sum(self._ball_votes) > len(self._ball_votes) // 2:
            self._offense_cluster ^= 1
            self._ball_votes.clear()
            self._flip_cooldown = self._VOTE_WIN  # prevent immediate re-flip
            print(f"team_classifier: offense cluster flipped → {self._offense_cluster}")
            return True

        return False


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

    detector   = YOLO(model_path)
    classifier = TeamClassifier()
    fitted     = False

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
                if cls == 0:
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

        cv2.putText(frame, f"Frame {frame_num}  offense=cyan  defense=orange",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Team Classifier", frame)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()
