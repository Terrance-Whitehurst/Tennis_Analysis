# Task: Replace TrackNet Ball Tracking with RF-DETR and Clean Up

## Context

The repo (`Tennis_Analysis`) currently uses **TrackNetV3 + InpaintNet** (heatmap U-Net + 1D trajectory inpainter) as the ball-tracking pipeline. A SageMaker training job was added later that trains an **RF-DETR** object detector (`scripts/sagemaker/entry_ball_detection.py`) on a COCO-format tennis ball dataset, producing a `model.pt` checkpoint uploaded to S3. TrackNet is now redundant — we want to rip it out entirely and replace the ball-tracking inference path with an RF-DETR-based detector that consumes the SageMaker checkpoint.

End state: the demo pipeline annotates the ball using RF-DETR. TrackNet code, tests, weights, and docs are gone. `pytest` and `make` targets still pass. Changes pushed to `main`.

## Scope of Work

### 1. Add a new RF-DETR inference wrapper

Create `src/inference/ball_tracking.py` (replacing the existing TrackNet-based file). It should:

- Load an RF-DETR checkpoint via `rfdetr.RFDETRBase` / `RFDETRLarge` (match the `--model` arg used at training time in `scripts/sagemaker/entry_ball_detection.py:108`). Inspect the `rfdetr` package API — do not guess method signatures.
- Expose a class/function with roughly the same public signature as the current `BallTracker` so the demo pipeline does not need invasive changes. Check how `src/inference/predict.py` and the demo script (grep for current `ball_tracking` imports) consume it today and preserve that contract: per-frame `(x, y, visibility)` or equivalent.
- Run inference frame-by-frame on video input (no 8-frame temporal stacking — RF-DETR is a per-image detector). For multi-detection frames, pick highest-confidence ball. If confidence below threshold, mark frame as no-ball.
- Keep the existing fading trajectory-trail visualization (that logic currently lives in `src/utils/visualize.py` — move to a ball-specific helper if it's TrackNet-coupled, otherwise leave alone).
- Add a CLI arg `--rfdetr-model` (path to checkpoint) replacing `--tracknet-model` / `--inpaintnet-model`.

### 2. Delete TrackNet code

Remove these files outright:
- `src/models/tracknet.py`
- `src/datasets/tracknet_dataset.py`
- `src/training/train_tracknet.py`
- `src/inference/predict.py` (TrackNet-specific; verify nothing else uses it first)
- `src/evaluation/evaluate.py` — delete the file if `eval_tracknet` / `eval_inpaintnet` are its only contents; otherwise strip just those functions
- `tests/test_models.py` (entirely TrackNet/InpaintNet)
- TrackNet-specific test cases in `tests/test_utils.py` (keep non-TrackNet tests)
- `scripts/generate_mask_data.py` (TrackNet→InpaintNet mask generation)

For these scripts that `import src.datasets.tracknet_dataset`, decide per-file whether the script is still useful without TrackNet; if yes, refactor the import; if no, delete:
- `scripts/preprocess.py`
- `scripts/correct_label.py`
- `scripts/error_analysis.py`

Update `src/datasets/__init__.py` to drop the TrackNet dataset export.

### 3. Delete weight files if present

Check `models/` directory for:
- `TrackNet_best.pt`
- `InpaintNet_best.pt`
- Any `TrackNetV3_ckpts*` artifacts

Delete them from disk if found. They should already be gitignored (`data/` and `models/` typically are in CCDS layout) — confirm via `git ls-files models/` before assuming.

### 4. Strip TrackNet helpers from shared utils

Audit these for TrackNet-specific code:
- `src/utils/visualize.py`
- `src/utils/general.py`
- `src/utils/metric.py`

For each TrackNet-specific helper:
- If **unused** after the deletions above → remove it
- If **still used** by non-TrackNet code → leave it but rename/generalize if the name references TrackNet
- Verify with `grep -rn <function_name> src/ scripts/ tests/` before deleting

### 5. Update Makefile and README

`Makefile`:
- Delete the `train-tracknet` target
- Rewrite the `evaluate` target to point at RF-DETR (or delete if no eval pipeline exists yet)
- Remove `train-tracknet` from `.PHONY`
- Keep `train-ball-detection` (that's the RF-DETR SageMaker launcher)

`README.md`:
- Replace the "Ball Tracking: TrackNetV3" section with an RF-DETR description
- Remove the gdown/TrackNetV3 weight download instructions
- Update the demo command to use `--rfdetr-model` instead of `--tracknet-model` / `--inpaintnet-model`
- Remove the "Training — Ball Tracking (TrackNet)" section; keep or replace with RF-DETR SageMaker instructions
- Remove the TrackNetV3 citation from the bottom if no longer used

### 6. Verify nothing is broken

Before committing:
- `uv run pytest` passes (all remaining tests)
- `uv run python -c "import src.inference.ball_tracking"` imports clean
- `make help` shows the updated target list without errors
- Grep for any stragglers: `grep -rni -e tracknet -e inpaintnet src/ scripts/ tests/ Makefile README.md` should return nothing (or only intentional historical mentions).

### 7. Commit and push

- Single commit (or logical commits if the diff is huge): `remove TrackNet/InpaintNet ball tracking in favor of RF-DETR`
- Push directly to `main` (the user has authorized this)
- Do **not** use `--no-verify` or amend published commits

## Constraints

- Use `uv` for all Python operations (`uv run`, `uv add`, `uv sync`). The repo follows CCDS v2 structure.
- Do not guess the `rfdetr` inference API — read the installed package or its docs.
- Raw data is immutable; don't touch `data/raw/`.
- If you discover the RF-DETR checkpoint path convention is unclear (e.g., no trained checkpoint available locally to test against), flag it rather than inventing behavior.
- If a TrackNet helper turns out to be tangled with code that's still needed, surface the tradeoff instead of silently copying the logic.

## Deliverable

A pushed commit on `main` plus a short report listing: files deleted, files added, files modified, test results, and any decisions that needed judgment calls (e.g., scripts you deleted vs. refactored, utility helpers you kept vs. removed).
