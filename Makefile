.PHONY: install convert-yolo convert-court-seg train-player-detection train-court-keypoint train-ball-detection train-court-segmentation-modal inference test-models test clean help pull-models

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

## Convert COCO annotations to YOLO format (court keypoints)
convert-yolo:
	$(PYTHON_INTERPRETER) scripts/convert_coco_to_yolo_kpt.py

## Convert court segmentation COCO annotations to YOLO format
convert-court-seg:
	$(PYTHON_INTERPRETER) scripts/convert_coco_to_yolo_seg.py

## Train RF-DETR player detection model
train-player-detection:
	$(PYTHON_INTERPRETER) -m src.training.train_player_detection

## Train YOLO-Pose court keypoint model
train-court-keypoint:
	$(PYTHON_INTERPRETER) -m src.training.train_court_keypoint

## Train RF-DETR tennis ball detection model
train-ball-detection:
	$(PYTHON_INTERPRETER) -m src.training.train_ball_detection

## Train court segmentation model on Modal (serverless GPU)
train-court-segmentation-modal:
	$(PYTHON_INTERPRETER) scripts/modal/launch_court_segmentation.py

## Run RF-DETR ball tracking inference on test video
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
