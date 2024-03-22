from omegaconf import OmegaConf
from pprint import pp

from lpn.inverse_celeba import main_celeba

# set parameters manually
args = OmegaConf.create()
args.dataset_config = OmegaConf.create()
args.dataset_config.dataset = "celeba"
args.dataset_config.root = "data/celeba"
args.dataset_config.start_idx = 100
args.dataset_config.num_imgs = 20

args.operator_config = OmegaConf.create()
args.operator_config.operator = "blur"
args.operator_config.sigma_blur = 1.0
args.operator_config.image_size = 128
args.sigma_noise = 0.02

args.prox_config = OmegaConf.create()
args.prox_config.prox = "lpn"
args.prox_config.model_path = "exps/celeba/models/lpn/s=0.05/model.pt"

args.admm_config = OmegaConf.create()
args.admm_config.rho = 0.1
args.admm_config.maxiter = 20
args.admm_config.x0 = "adjoint"
args.admm_config.scale = 0.5
args.admm_config.order = "132"


args.model_config = OmegaConf.create()
args.model_config.model = "lpn_128"
args.model_config.params = OmegaConf.create()
args.model_config.params.in_dim = 3
args.model_config.params.hidden = 256
args.model_config.params.beta = 100
args.model_config.params.alpha = 1e-6


args.seed = 0
args.out_dir = None
args.measure = False
args.solver = "admm"
args.data_dir = None


# set commandline parameters
def set_cmd(args):
    """Set the input parameters from commandline"""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--sigma_blur", type=float, required=True)
    parser.add_argument("--sigma_noise", type=float, required=True)
    args_cmd = parser.parse_args()
    pp(args_cmd)
    args.operator_config.sigma_blur = args_cmd.sigma_blur
    args.sigma_noise = args_cmd.sigma_noise
    args.out_dir = f"exps/celeba/results/inverse/deblur/blur={args_cmd.sigma_blur}_noise={args_cmd.sigma_noise}/admm"
    return args


args = set_cmd(args)


# set best parameters
def set_best(args):
    """Set the best parameters for each setting"""
    if args.operator_config.sigma_blur == 1.0 and args.sigma_noise == 0.02:
        args.prox_config.model_path = "exps/celeba/models/lpn/s=0.05/model.pt"
        args.admm_config.scale = 0.5
    elif args.operator_config.sigma_blur == 1.0 and args.sigma_noise == 0.04:
        args.prox_config.model_path = "exps/celeba/models/lpn/s=0.1/model.pt"
        args.admm_config.scale = 0.5
    elif args.operator_config.sigma_blur == 2.0 and args.sigma_noise == 0.02:
        args.prox_config.model_path = "exps/celeba/models/lpn/s=0.1/model.pt"
        args.admm_config.scale = 2.0
    elif args.operator_config.sigma_blur == 2.0 and args.sigma_noise == 0.04:
        args.prox_config.model_path = "exps/celeba/models/lpn/s=0.1/model.pt"
        args.admm_config.scale = 0.5
    else:
        raise ValueError("Best parameters not defined for the given setting")
    return args


args = set_best(args)


pp(args)
main_celeba(args)
