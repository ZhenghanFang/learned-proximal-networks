python lpn/train.py \
--exp_dir exps/mnist/experiments/mnist \
--dataset_config_path exps/mnist/configs/mnist/dataset.json \
--model_config_path exps/mnist/configs/mnist/model.json \
--train_batch_size 200 \
--dataloader_num_workers 8 \
--num_steps 40000 \
--num_steps_pretrain 20000 \
--pretrain_lr 1e-3 \
--lr 1e-4 \
--num_stages 4 \
--image_size 28 \
--num_channels 1 \
--validate_every_n_steps 100
