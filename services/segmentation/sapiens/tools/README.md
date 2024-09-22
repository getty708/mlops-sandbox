# Tools

## Demo: Run the sapiens segmentation on a single image. [`run_inference.py`](./run_inference.py)

See the [`../README.md`](../README.md) for more details.

## Demo Script from the Original Repository (`lite/demo/vis_seg.py`)

Reference: https://github.com/facebookresearch/sapiens/blob/main/lite/demo/vis_seg.py

- Download the model checkpoint from the Sapiens repository.

```bash
$ cd services/segmentation/sapiens
$ make download
```

- Prepare input images.

```
$ cd services/segmentation/sapiens/tools
$ tree outputs
outputs
└── inputs
    └── bbox_0.png
```

- Run the script.

```bash
$ cd services/segmentation/sapiens/tools
$ python vis_seg.py \
	../data/models/sapiens_0.3b_goliath_best_goliath_mIoU_7673_epoch_194_torchscript.pt2 \
	--input ./outputs/inputs/ \
	--output_root ./outputs/outputs/
$ tree outputs
outputs
├── inputs
│   └── bbox_0.png
└── outputs
    ├── bbox_0.npy
    ├── bbox_0.png
    └── bbox_0_seg.npy
```
