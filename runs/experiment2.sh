
# d24 model (slightly overtrained is enough to beat GPT-2 => increase data:params ratio from compute optimal 10.5 (default) to 12)
torchrun --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=20 --target-param-data-ratio=12 --device-batch-size=16 --run=e2 --drop-loss-start=0.10 --drop-loss-end=0.0 --drop-loss-decay-ratio=0.5 --drop-loss-warmup-ratio=0.00
# evaluate the model: CORE metric, BPB on train/val, and draw samples
torchrun --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16