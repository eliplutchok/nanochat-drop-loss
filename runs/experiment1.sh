#!/bin/bash
source .venv/bin/activate
# this is the baseline
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --target-param-data-ratio=12 --device-batch-size=16 --run=e1
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16