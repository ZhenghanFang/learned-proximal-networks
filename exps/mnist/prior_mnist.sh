
for perturb in convex blur gaussian; do
    python lpn/evaluate_prior.py \
    --model_config_path exps/mnist/configs/mnist/model.json \
    --out_dir exps/mnist/experiments/mnist/prior \
    --perturb_config_path exps/mnist/configs/mnist/prior/perturb/${perturb}.json \
    --dataset_config_path exps/mnist/configs/mnist/prior/dataset.json \
    --model_path exps/mnist/experiments/mnist/model.pt \
    --inv_alg cvx_cg 
done
