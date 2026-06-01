# Benchmark Label Format

Label the PNGs in `key_frames/bijon_run/`, then save the labels in
`benchmarks/fixtures/bijon_run_labels.json`.

Player boxes are required for detection accuracy. Team, jersey number, and ball
labels are optional; metrics for missing optional labels are skipped.

```json
{
  "video_path": "test media/videos/bijon_run.mp4",
  "frames": [
    {
      "frame_number": 1,
      "image_path": "benchmarks/fixtures/key_frames/bijon_run/frame_000001.png",
      "players": [
        {
          "bbox": [100, 200, 150, 310],
          "team": "offense",
          "number": "12"
        }
      ],
      "ball": {
        "center": [850, 420]
      }
    }
  ]
}
```

Use pixel coordinates in the exported PNG's original resolution. Omit `team`,
`number`, or `ball` when you do not want that frame to count for the metric.
