# 49ers Playbook Reconstruction - Implementation Plan

## Core Concept
Watch broadcast football footage → reconstruct each play as a top-down diagram → save to build a playbook.

---

## Step 1: Player Detector
**Goal**: Find all players in the broadcast frame.

- [ ] Use RF-DETR S (fine-tune on football frames)
- [ ] Two-resolution inference for bunched players at LOS
- [ ] Output: bounding boxes for all 22 players per frame

---

## Step 2: Tracker
**Goal**: Keep each player's identity across frames.

- [ ] Implement ByteTrack or DeepSORT
- [ ] Assign consistent ID to each player throughout the play
- [ ] Handle occlusions when players cross paths
- [ ] Output: player ID + position for every frame

---

## Step 3: Field Mapping / Homography
**Goal**: Convert broadcast camera coordinates into top-down field coordinates.

- [ ] Detect yard lines using Hough transform
- [ ] Compute vanishing point VP from pairwise line intersections
- [ ] Find 4+ correspondence points (yard lines, hash marks)
- [ ] Compute homography H ∈ ℝ³ˣ³ mapping (u,v) → (x,y) yards
- [ ] Ray-cast from player feet toward VP for depth
- [ ] Output: (x, y) field coordinates for each player

---

## Step 4: Team Classifier
**Goal**: Separate offense vs defense by jersey color.

- [ ] Extract jersey crop from each bounding box
- [ ] Mask out grass (green pixels)
- [ ] K-means (k=2) on jersey colors
- [ ] Lock centroids on frame 1, match by L2 distance
- [ ] Output: team label (offense/defense) per player

---

## Step 5: Jersey Number Recognizer
**Goal**: Label players by jersey number.

- [ ] Crop jersey number region from bounding box
- [ ] OCR or small CNN classifier for digits 0-99
- [ ] Track specific players by name using roster lookup
- [ ] Output: jersey number per player

---

## Step 6: Sideline Detection (Fallback)
**Goal**: Handle when painted sideline isn't visible.

- [ ] Primary: detect painted sideline via edge detection
- [ ] Fallback: DBSCAN (ε=40px, minPts=25) on player feet
- [ ] Fit least-squares line as proxy sideline
- [ ] Filter out refs, coaches, crowd

---

## Step 7: Ball Tracking
**Goal**: Track ball to reconstruct the full route.

- [ ] Detect ball as separate class
- [ ] Track trajectory post-snap with Kalman filter
- [ ] Link throw origin → QB position
- [ ] Link catch point → nearest receiver
- [ ] Output: ball trajectory + throw/catch endpoints

---

## Step 8: Play Segmentation
**Goal**: Automatically detect when plays start and end.

- [ ] Detect snap: sudden coordinated motion from offense
- [ ] Detect play end: whistle, tackle, out of bounds, incompletion
- [ ] Segment continuous video into individual plays
- [ ] Output: frame ranges for each play

---

## Step 9: Real-Time Reconstruction
**Goal**: Build top-down view as play unfolds.

- [ ] Initialize 2D field canvas (100 × 53.3 yards)
- [ ] Each frame: update player positions from homography
- [ ] Trace routes (cumulative path history per player)
- [ ] Draw ball trajectory
- [ ] Render in real-time as play progresses

---

## Step 10: Play Diagram Export
**Goal**: Save the final reconstructed play.

- [ ] At play end, generate diagram showing:
  - Pre-snap formation (where everyone lined up)
  - All routes traced (offensive player paths)
  - Ball trajectory (throw arc)
  - Final positions
- [ ] Export as PNG and structured JSON
- [ ] Store with metadata: game, quarter, down, distance, result
- [ ] Build searchable playbook over time

---

## Component Summary

| Component | Input | Output |
|-----------|-------|--------|
| Player Detector | Frame | Bounding boxes |
| Tracker | Boxes + previous frame | Player IDs |
| Field Mapping | Lines + boxes | (x,y) field coords |
| Team Classifier | Jersey crops | Offense/defense label |
| Jersey OCR | Number crop | Jersey number |
| Ball Tracking | Frames | Ball trajectory |
| Play Segmenter | Video | Frame ranges |
| Reconstructor | All above | Top-down animation |
| Diagram Export | Final state | PNG + JSON |

---

## Tech Stack
- **Detection**: RF-DETR S
- **Tracking**: ByteTrack
- **CV**: OpenCV
- **Homography**: cv2.findHomography
- **Clustering**: scikit-learn (K-means, DBSCAN)
- **Visualization**: matplotlib, Pillow

---

## Team Breakdown

### Person 1: Detection & Tracking
- Step 1: Player Detector (RF-DETR S)
- Step 2: Tracker (ByteTrack)
- Step 7: Ball Tracking

Core skills: Deep learning, object detection, tracking algorithms. These components are tightly coupled - tracker depends on detector output, ball tracking uses similar techniques.

---

### Person 2: Field Geometry & Segmentation
- Step 3: Field Mapping / Homography will need to train a keypont model like in ai basketball video
- Step 6: Sideline Detection (fallback)
- Step 8: Play Segmentation

Core skills: Classical CV, camera geometry, signal processing. All about understanding the field coordinate system and temporal boundaries of plays.

---

### Person 3: Classification & Visualization
- Step 4: Team Classifier
- Step 4 also: Jersey Number Recognizer
- Step 9: Real-Time Reconstruction
- Step 10: Play Diagram Export

Core skills: Image classification, OCR, visualization/rendering. Takes detection outputs and turns them into meaningful labeled diagrams.

---

### Integration Points
- Person 1 delivers bounding boxes → Person 2 needs them for homography, Person 3 needs them for classification
- Person 2 delivers field coordinates → Person 3 needs them for reconstruction
- Final integration: Person 3's reconstruction consumes all outputs
B