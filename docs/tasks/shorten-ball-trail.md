# Task: Tighten the Ball Trail Annotation in `test_models_on_video.py`

## Context

`scripts/test_models_on_video.py` renders annotated MP4s for the demo pipeline (player + ball + court-segmentation models). The ball annotation today is a cyan trail (`sv.TraceAnnotator`) plus a small triangle marker on the current frame.

The trail currently lags well behind the ball and has a long, slow tail — visually it feels like the trail is dragging behind the live position rather than tracking it. The desired look is a **very short, snappy trail that sits right behind the ball as it moves**, more like a comet glint than a long streak. The triangle marker on the current frame should remain.

## Root Cause

Two settings in `scripts/test_models_on_video.py` (lines around 219–226) compound to produce the lag:

1. **`trace_length=int(fps * 1.5)`** — at 50 FPS this is **75 frames ≈ 1.5 seconds of trail history**. That's far too long.
2. **`ball_smoother = sv.DetectionsSmoother(length=5)`** — averages the ball position over the last 5 frames before drawing. This adds a perceptible head-of-trail lag (~50–100 ms at 50 FPS), which is the main reason the marker feels "behind" the ball.

The combination: smoother shifts the head backward in time, then the long `trace_length` paints a slow-fading streak over that already-late position.

## Scope of Work

### 1. Shorten the trail length

In `scripts/test_models_on_video.py`, locate the `ball_trace_annotator` definition (currently lines 219–225):

```python
ball_trace_annotator = sv.TraceAnnotator(
    color=sv.ColorPalette.from_hex(["#00FFFF"]),
    position=sv.Position.CENTER,
    trace_length=int(fps * 1.5),   # ← change this
    thickness=2,
    color_lookup=sv.ColorLookup.TRACK,
)
```

Change `trace_length` to a small fixed value that resolves to ~80–120 ms of history regardless of FPS:

```python
trace_length=max(4, int(fps * 0.1)),   # ~0.1 s, floor of 4 frames
```

At 50 FPS this evaluates to 5 frames; at 25 FPS to 4 frames. Keep `thickness=2` (the trail should be subtle but visible). If after eyeballing the result it still looks too long, drop the multiplier to `0.08` or hard-code `trace_length=4`.

### 2. Remove the smoother from the ball path

In the same file, line 226:

```python
ball_smoother = sv.DetectionsSmoother(length=5)
```

…and its use inside the per-frame loop (currently around line 340):

```python
best_det = ball_smoother.update_with_detections(best_det)
```

Delete both. Justification: the ball is a single high-confidence detection per frame and ByteTrack-style smoothing is unnecessary; what the smoother actually does here is delay the head of the trail. The ball detector already runs at >70 % per-frame recall on the test clips, so jitter is not a real problem.

If — after deletion — the trail looks too jumpy on missed-detection frames, reintroduce a length-2 smoother (`sv.DetectionsSmoother(length=2)`); do **not** revert to length 5.

### 3. Re-render the demo videos and verify

Run the pipeline against the three test clips:

```bash
uv run python scripts/test_models_on_video.py
```

Outputs land at `reports/figures/Clip{1,2,3}_annotated.mp4`. Visually verify on at least Clip1 and Clip3 (which have the fastest ball motion):

- The trail should be **roughly the length of the ball's motion over ~3–5 frames** — about a racket-head's worth on a typical baseline shot, not a long streak.
- The marker triangle should sit **on** the ball, not lagging behind it. Pause on a few high-velocity frames in a video player and confirm the triangle and the ball overlap.
- Frame counts and ball-detection percentages reported at the end of each clip should match the previous run (changes are visual only — detection logic is untouched).

If detection rates regress or the script raises, the smoother removal was misapplied — re-check that `best_det` is still passed unchanged into `ball_trace_annotator.annotate()` and `ball_triangle_annotator.annotate()`.

### 4. Optional polish (skip if time-boxed)

`sv.TraceAnnotator` does not natively taper the trail (no per-segment thickness or alpha gradient). If a comet-style fade is desired later, that's a custom annotator and a separate task — not in scope here.

## Files Touched

- `scripts/test_models_on_video.py` — two edits in the `process_video()` function (annotator config + per-frame loop). No other files should change.

## Acceptance Criteria

- [ ] `trace_length` no longer references `fps * 1.5`; new value resolves to ≤ 8 frames at 50 FPS.
- [ ] `ball_smoother` and its `.update_with_detections(best_det)` call are deleted (or smoother length is ≤ 2 with explicit justification).
- [ ] All three clips re-rendered into `reports/figures/Clip{1,2,3}_annotated.mp4`.
- [ ] Visual review confirms the trail is short and the triangle tracks the ball with no perceptible lag.
- [ ] Per-clip detection stats (frames, players/frame, court %, ball %) are within ±2 % of the prior run.
