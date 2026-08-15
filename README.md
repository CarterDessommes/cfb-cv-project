# CFB CV Project

College Football Computer Vision Project

## Benchmarks

Export the frames that need labels:

```bash
python3 benchmarks/run_benchmarks.py --get-key-frames
```

Label the exported PNGs in `benchmarks/fixtures/key_frames/bijon_run/`, then
fill `benchmarks/fixtures/bijon_run_labels.json` using the format documented in
`benchmarks/fixtures/LABELING.md`. Frame numbers are zero-based and match the
PNG filenames.

Run the benchmark check:

```bash
python3 benchmarks/run_benchmarks.py --check
```

If labels were exported from Roboflow as COCO JSON, convert them first:

```bash
python3 benchmarks/run_benchmarks.py --import-coco benchmarks/fixtures/roboflow/keyframes_coco/train/_annotations.coco.json
```

Run the passing-video benchmark:

```bash
python3 benchmarks/run_benchmarks.py --benchmark pass1 --check
```

The first successful labeled run creates `benchmarks/baseline.json`. Later
checks fail on speed or accuracy regressions, and automatically update the
baseline when the run passes all gates and improves a tracked metric.

Timings in a baseline are only comparable on the machine/session that recorded
them. After a hardware change (or when a stale baseline's FPS/p95 gates fail on
code that hasn't changed), delete the baseline JSON and re-run `--check` once to
re-baseline; accuracy metrics are deterministic and comparable everywhere.
