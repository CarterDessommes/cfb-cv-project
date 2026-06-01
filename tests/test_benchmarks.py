import json
from pathlib import Path

import cv2
import numpy as np

from benchmarks import run_benchmarks


def test_bbox_iou():
    assert run_benchmarks.bbox_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert run_benchmarks.bbox_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert round(run_benchmarks.bbox_iou([0, 0, 10, 10], [5, 5, 15, 15]), 3) == 0.143


def test_compute_accuracy_metrics_skips_missing_optional_labels():
    labels = {
        "frames": [
            {
                "frame_number": 1,
                "players": [
                    {"bbox": [0, 0, 10, 10], "team": "offense", "number": "12"},
                    {"bbox": [20, 20, 30, 30]},
                ],
            }
        ]
    }
    predictions = {
        "predictions": [
            {
                "frame_number": 1,
                "players": [
                    {"bbox": [0, 0, 10, 10], "team": "offense", "number": "12"},
                    {"bbox": [20, 20, 30, 30], "team": "defense", "number": "80"},
                    {"bbox": [80, 80, 90, 90], "team": "defense", "number": "1"},
                ],
                "ball": None,
            }
        ]
    }

    metrics = run_benchmarks.compute_accuracy_metrics(labels, predictions)

    assert metrics["player_true_positives"] == 2
    assert metrics["player_false_positives"] == 1
    assert metrics["player_false_negatives"] == 0
    assert round(metrics["player_f1"], 3) == 0.800
    assert metrics["team_accuracy"] == 1.0
    assert metrics["team_labeled_count"] == 1
    assert metrics["jersey_accuracy"] == 1.0
    assert metrics["jersey_labeled_count"] == 1
    assert metrics["ball_hit_rate"] is None


def test_compare_to_baseline_fails_on_speed_regression():
    baseline = {
        "timings": {"fps": 100.0, "p95_frame_seconds": 0.01},
        "accuracy": {
            "player_f1": 0.9,
            "team_accuracy": None,
            "jersey_accuracy": None,
            "ball_hit_rate": None,
        },
    }
    current = {
        "timings": {"fps": 80.0, "p95_frame_seconds": 0.01},
        "accuracy": {
            "player_f1": 0.9,
            "team_accuracy": None,
            "jersey_accuracy": None,
            "ball_hit_rate": None,
        },
    }

    passed, improved, failures = run_benchmarks.compare_to_baseline(current, baseline)

    assert passed is False
    assert improved is False
    assert any("FPS regressed" in failure for failure in failures)


def test_compare_to_baseline_detects_clean_improvement():
    baseline = {
        "timings": {"fps": 100.0, "p95_frame_seconds": 0.01},
        "accuracy": {
            "player_f1": 0.9,
            "team_accuracy": 0.8,
            "jersey_accuracy": None,
            "ball_hit_rate": None,
        },
    }
    current = {
        "timings": {"fps": 105.0, "p95_frame_seconds": 0.01},
        "accuracy": {
            "player_f1": 0.9,
            "team_accuracy": 0.8,
            "jersey_accuracy": None,
            "ball_hit_rate": None,
        },
    }

    passed, improved, failures = run_benchmarks.compare_to_baseline(current, baseline)

    assert passed is True
    assert improved is True
    assert failures == []


def test_extract_key_frames_creates_images_and_label_template(tmp_path, monkeypatch):
    video_path = tmp_path / "tiny.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10,
        (16, 16),
    )
    for value in [0, 80, 160]:
        writer.write(np.full((16, 16, 3), value, dtype=np.uint8))
    writer.release()

    label_doc = tmp_path / "fixtures" / "LABELING.md"
    monkeypatch.setattr(run_benchmarks, "LABELING_DOC", label_doc)
    output_dir = tmp_path / "frames"
    labels_path = tmp_path / "labels.json"

    written = run_benchmarks.extract_key_frames(video_path, [0, 2], output_dir, labels_path)

    assert [path.name for path in written] == ["frame_000000.png", "frame_000002.png"]
    assert all(path.exists() for path in written)
    assert label_doc.exists()
    labels = json.loads(labels_path.read_text())
    assert [frame["frame_number"] for frame in labels["frames"]] == [0, 2]
    assert labels["frames"][0]["players"] == []


def test_cli_check_uses_fake_pipeline_and_creates_baseline(tmp_path, monkeypatch):
    labels_path = tmp_path / "labels.json"
    labels_path.write_text(json.dumps({
        "frames": [
            {
                "frame_number": 1,
                "players": [{"bbox": [0, 0, 10, 10]}],
            }
        ]
    }))

    def fake_pipeline(**_kwargs):
        return {
            "timings": {
                "model_load_seconds": 0.1,
                "processing_seconds": 0.2,
                "total_seconds": 0.3,
                "fps": 10.0,
                "mean_frame_seconds": 0.1,
                "p95_frame_seconds": 0.1,
            },
            "predictions": [
                {
                    "frame_number": 1,
                    "players": [{"bbox": [0, 0, 10, 10], "team": "unknown", "number": "?"}],
                    "ball": None,
                }
            ],
        }

    monkeypatch.setattr(run_benchmarks, "run_pipeline_benchmark", fake_pipeline)
    baseline = tmp_path / "baseline.json"
    results_dir = tmp_path / "results"

    exit_code = run_benchmarks.main([
        "--check",
        "--labels", str(labels_path),
        "--baseline", str(baseline),
        "--results-dir", str(results_dir),
    ])

    assert exit_code == 0
    assert baseline.exists()
    assert list(results_dir.glob("benchmark_*.json"))
