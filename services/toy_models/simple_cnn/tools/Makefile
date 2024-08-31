MODEL_NUM_LAYERS := 3

OUTPUT_DIR := ./outputs
ONNX_FILENAME := simple_cnn_l${MODEL_NUM_LAYERS}.onnx


.PHONY: build
build:
	-@mkdir -p ${OUTPUT_DIR}
	python convert_to_onnx.py --onnx-path ${OUTPUT_DIR}/${ONNX_FILENAME} --num-layers ${MODEL_NUM_LAYERS}

.PHONY: build-torchscript
build-torchscript:
	-@mkdir -p ${OUTPUT_DIR}
	python convert_to_onnx_torchscript.py \
		--onnx-path ${OUTPUT_DIR}/${ONNX_FILENAME} \
		--num-layers ${MODEL_NUM_LAYERS}

.PHONY: clean
clean:
	rm -rf *.onnx
