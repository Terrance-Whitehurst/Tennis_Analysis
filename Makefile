.PHONY: install data train-tracknet train-player-detection train-court-keypoint evaluate inference test clean help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))
PROJECT_NAME = tennis-analysis
PYTHON_INTERPRETER = uv run python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
install:
	uv pip install -e .

## Preprocess raw data
data:
	$(PYTHON_INTERPRETER) scripts/preprocess.py

## Convert COCO annotations to YOLO format (court keypoints)
convert-yolo:
	$(PYTHON_INTERPRETER) scripts/convert_coco_to_yolo_kpt.py

## Train TrackNet ball tracking model
train-tracknet:
	$(PYTHON_INTERPRETER) -m src.training.train_tracknet \
		--model_name TrackNet \
		--seq_len 8 \
		--epochs 30 \
		--batch_size 10 \
		--bg_mode concat \
		--save_dir experiments/tracknet_v1

## Train RF-DETR player detection model
train-player-detection:
	$(PYTHON_INTERPRETER) -m src.training.train_player_detection

## Train YOLO-Pose court keypoint model
train-court-keypoint:
	$(PYTHON_INTERPRETER) -m src.training.train_court_keypoint

## Evaluate TrackNet on test split
evaluate:
	$(PYTHON_INTERPRETER) -m src.evaluation.evaluate \
		--tracknet_file models/TrackNet_best.pt \
		--split test \
		--eval_mode weight

## Run ball tracking inference on test video
inference:
	$(PYTHON_INTERPRETER) -m src.inference.ball_tracking

## Run test inference for all models on test video
test-models:
	$(PYTHON_INTERPRETER) scripts/test_models_on_video.py

## Run unit tests
test:
	uv run pytest tests/ -v

## Delete compiled Python files and caches
clean:
	find . -type f -name "*.py[cod]" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +

## Pull model artifacts from S3
pull-models:
	mkdir -p models/player_detection models/court_keypoint
	aws s3 cp s3://training-jobs-test-315109499400/tennis-analysis/models/player_detection/rfdetr-player-base-20260403-054623/output/model.tar.gz models/player_detection/model.tar.gz
	aws s3 cp s3://training-jobs-test-315109499400/tennis-analysis/models/court_keypoint/yolo-court-yolo11m-20260403-070739/output/model.tar.gz models/court_keypoint/model.tar.gz
	tar xzf models/player_detection/model.tar.gz -C models/player_detection/
	tar xzf models/court_keypoint/model.tar.gz -C models/court_keypoint/

#################################################################################
# Self-documenting Makefile                                                     #
#################################################################################

## Show this help message
help:
	@echo "$$(tput bold)Available commands:$$(tput sgr0)"
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_hierarchical=6 \
		'{ \
			printf "%s%*s ", $$1, indent - length($$1), ""; \
			n = split($$2, words, " "); \
			line_length = ncol - indent; \
			for (i = 1; i <= n; i++) { \
				line_length -= length(words[i]) + 1; \
				if (line_length <= 0) { \
					line_length = ncol - indent - length(words[i]) - 1; \
					printf "\n%*s ", indent, " "; \
				} \
				printf "%s ", words[i]; \
			} \
			printf "\n"; \
		}' \
	| more

.DEFAULT_GOAL := help
