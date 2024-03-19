#!/bin/bash

for NOISE in 0.1; do
    python lpn/train.py \
    --exp_dir exps/mayoct/models/lpn/n=${NOISE} \
    --dataset_config_path exps/mayoct/configs/mayoct/dataset.json \
    --model_config_path exps/mayoct/configs/mayoct/model.json \
    --train_batch_size 64 \
    --dataloader_num_workers 4 \
    --num_steps 40000 \
    --num_steps_pretrain 20000 \
    --pretrain_lr 1e-3 \
    --lr 1e-4 \
    --num_stages 4 \
    --image_size 128 \
    --num_channels 1 \
    --sigma_noise ${NOISE}
done