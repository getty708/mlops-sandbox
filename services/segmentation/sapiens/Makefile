MODEL_ROOT_DIR := ./data/models
MODEL_SIZSE := 0.6b

HG_MODEL_CARD_NAME := facebook/sapiens-seg-${MODEL_SIZSE}-torchscript

.PHONY: download
download:
	huggingface-cli download ${HG_MODEL_CARD_NAME} --local-dir ${MODEL_ROOT_DIR}/${HG_MODEL_CARD_NAME}
