#  this is the ablation study of drop-loss
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --target-param-data-ratio=12 --device-batch-size=16 --run=e3 --drop-loss-start=0.10 --drop-loss-end=0.0 --drop-loss-decay-ratio=0.5 --drop-loss-warmup-ratio=0.00 --drop-loss-random
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16