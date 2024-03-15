NOISE=0.001

# Measure
for M in 16384 65536; do
    python exps/mayoct/inverse_mayoct_cs.py \
    --measure \
    --dataset_config_path exps/mayoct/configs/mayoct/inverse/dataset.json \
    --operator_config_path exps/mayoct/configs/mayoct/inverse/cs/operator_cs_M=${M}.json \
    --sigma_noise ${NOISE} \
    --out_dir exps/mayoct/results/inverse/mayoct/cs/M=${M}_noise=${NOISE}
done

# LPN
METHOD=lpn
for M in 16384 65536; do
    python exps/mayoct/inverse_mayoct_cs.py \
    --dataset_config_path exps/mayoct/configs/mayoct/inverse/dataset.json \
    --operator_config_path exps/mayoct/configs/mayoct/inverse/cs/operator_cs_M=${M}.json \
    --prox_config_path exps/mayoct/configs/mayoct/inverse/cs/prox_${METHOD}.json \
    --admm_config_path exps/mayoct/configs/mayoct/inverse/cs/admm.json \
    --sigma_noise ${NOISE} \
    --out_dir exps/mayoct/results/inverse/mayoct/cs/M=${M}_noise=${NOISE}
done
