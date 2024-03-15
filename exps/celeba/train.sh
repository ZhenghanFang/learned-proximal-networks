for NOISE in 0.05 0.1; do
    python lpn/train.py \
    --exp_dir exps/celeba/models/lpn/s=${NOISE} \
    --dataset_config_path exps/celeba/configs/dataset.json \
    --model_config_path exps/celeba/configs/model.json \
    --train_batch_size 64 \
    --dataloader_num_workers 8 \
    --num_steps 40000 \
    --num_steps_pretrain 20000 \
    --pretrain_lr 1e-3 \
    --lr 1e-4 \
    --num_stages 4 \
    --save_every_n_steps 1000 \
    --image_size 128 \
    --num_channels 3 \
    --sigma_noise ${NOISE}
done
