"""Solve inverse problems using LPN."""

import argparse
import os
import scico
from scico import functional, linop, loss
from scico.optimize.admm import ADMM, LinearSubproblemSolver
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from omegaconf import OmegaConf


from lpn.utils import get_model, get_imgs, load_config
from lpn.operators.blur import get_blur_A
from lpn.operators.cs import get_cs_A
from lpn.utils import scico as nonjit_scico
from lpn.utils.lpn_for_inv import lpn_denoise_patch
from lpn.utils.admm import ADMM


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        help=("Dataset config."),
    )
    parser.add_argument(
        "--operator_config_path",
        type=str,
        default=None,
        help="The config of the forward operator.",
    )
    parser.add_argument(
        "--prox_config_path",
        type=str,
        default=None,
        help="The config of the proximal step.",
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="The config of the LPN model.",
    )
    parser.add_argument(
        "--admm_config_path",
        type=str,
        default=None,
        help="The config of the ADMM solver.",
    )
    parser.add_argument("--sigma_noise", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out_dir", type=str, default="results/inverse")
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--solver", type=str, default="admm")
    parser.add_argument("--data_dir", type=str, default=None)
    args = parser.parse_args()

    # load config
    args.dataset_config = load_config(args.dataset_config_path)
    args.operator_config = load_config(args.operator_config_path)
    args.prox_config = load_config(args.prox_config_path)
    if args.prox_config is not None and args.prox_config.prox == "lpn":
        args.model_config = OmegaConf.create(
            {
                "model": "lpn_128",
                "params": {
                    "in_dim": 1,
                    "hidden": 256,
                    "beta": 100,
                    "alpha": 1e-06
                }
            }
        )
    if args.admm_config_path is not None:
        args.admm_config = load_config(args.admm_config_path)
    if args.data_dir is None:
        args.data_dir = args.out_dir

    return args


class HelperFunctional(scico.functional.Functional):
    has_eval = False
    has_prox = True

    def __init__(self, prox):
        super().__init__()
        self._prox = prox

    def prox(self, x, lam=1.0, **kwargs):
        return self._prox(x, lam=lam, **kwargs)


def measure(x_list, A, sigma_noise, seed):
    """Measure images with forward operator and add noise
    Inputs:
        x_list: list of images to be measured
        A: forward operator
        sigma_noise: standard deviation of noise
        seed: random seed
    Outputs:
        y_list: list of measurements
    """
    # set random seed
    np.random.seed(seed)

    y_list = []
    for x in x_list:
        y = A(x)
        noise = np.random.normal(0, 1, y.shape)
        y = y + sigma_noise * noise
        y_list.append(np.asarray(y).astype("float32"))
    return y_list


def load_model(model_config, model_path):
    """Load LPN model for testing"""
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model


def main_mayoct_cs(args):
    x_list = get_imgs(args.dataset_config)
    if args.measure:
        args.operator_config.init = True
        np.random.seed(args.seed)
        A = get_A(args.operator_config)
        y_list = measure(x_list, A, args.sigma_noise, args.seed)
        save_results(x_list, os.path.join(args.out_dir, "meas", "x"))
        save_results(y_list, os.path.join(args.out_dir, "meas", "y"))
        A.save_operator(args.operator_config.operator_path)
        print("Measurements saved.")
        return

    args.operator_config.init = False
    A = get_A(args.operator_config)

    y_list = [
        np.load(os.path.join(args.out_dir, "meas", "y", f"{i}.npy"))
        for i in range(len(x_list))
    ]

    # ls_list = [
    #     np.load(os.path.join(args.out_dir, "ls", "xhat", f"{i}.npy"))
    #     for i in range(len(x_list))
    # ]

    print("Computing LS solution...")
    ls_list = []
    for y in tqdm(y_list):
        ls_list.append(A.get_ls(y))

    # get LPN proximal step
    model = load_model(args.model_config, args.prox_config.model_path)
    prox = lambda x, lam=1.0, **kwargs: lpn_denoise_patch(
        np.asarray(x), model, args.prox_config.patch_size, args.prox_config.stride_size
    )

    # parse ADMM arguments
    rho = args.admm_config.rho  # ADMM penalty parameter
    maxiter = args.admm_config.maxiter  # number of ADMM iterations
    subprob_tol = args.admm_config.subprob_tol  # tolerance for subproblem
    subprob_maxiter = (
        args.admm_config.subprob_maxiter
    )  # maximum iteration for subproblem
    scale_f = args.admm_config.scale_f  # scale for data fidelity term
    zero_one = args.admm_config.zero_one  # clip to [0, 1] after each iteration

    psnr_list, ssim_list = [], []
    for i, y in enumerate(tqdm(y_list)):
        x0 = ls_list[i]  # initialize with ls
        x_shape = x0.shape

        f = loss.SquaredL2Loss(y=y, A=A, scale=scale_f)

        g = HelperFunctional(prox)

        if zero_one:
            g_list = [g, nonjit_scico.ZeroOneIndicator()]
        else:
            g_list = [g]
        C_list = [linop.Identity(x_shape) for _ in range(len(g_list))]
        rho_list = [rho] * len(g_list)

        solver = ADMM(
            f=f,
            g_list=g_list,
            C_list=C_list,
            rho_list=rho_list,
            x0=x0,
            maxiter=maxiter,
            subproblem_solver=nonjit_scico.LinearSubproblemSolver(
                cg_kwargs={"tol": subprob_tol, "maxiter": subprob_maxiter}
            ),
            itstat_options={"display": True},
            order="132",
        )
        xhat = solver.solve()
        xhat = np.asarray(xhat)
        print(f"PSNR: {skimage_psnr(x_list[i], xhat, data_range=1.)}")
        psnr_val = skimage_psnr(x_list[i], xhat, data_range=1.)
        ssim_val = skimage_ssim(x_list[i], xhat,  data_range=1.)
        print(
            f"PSNR: {psnr_val}",
            f"SSIM: {ssim_val}",
        )
        psnr_list.append(psnr_val)
        ssim_list.append(ssim_val)
        # save
        _save(xhat, os.path.join(args.out_dir, args.prox_config.prox, "xhat"), i)
        _save(y, os.path.join(args.out_dir, args.prox_config.prox, "y"), i)
        _save(x_list[i], os.path.join(args.out_dir, args.prox_config.prox, "x"), i)
        _save(ls_list[i], os.path.join(args.out_dir, args.prox_config.prox, "ls"), i)

    recon_log = (
        "average performance: "
        "PSNR {:.3f} +/- {:.3f}, SSIM {:.4f} +/- {:.4f}".format(
            np.mean(psnr_list),
            np.std(psnr_list),
            np.mean(ssim_list),
            np.std(ssim_list),
        )
    )
    print(recon_log)
    with open(os.path.join(args.out_dir, args.prox_config.prox, "xhat", "recon_log.txt"), "w") as f:
        f.write(recon_log)


def _save(data, dir, i):
    os.makedirs(dir, exist_ok=True)
    np.save(os.path.join(dir, f"{i}.npy"), data)


def get_x0(y, A, x0_type):
    if x0_type == "zero":
        x0 = np.zeros(A.adjoint(y).shape)
    elif x0_type == "adjoint":
        x0 = A.adj(y)
    elif x0_type == "copy":
        x0 = y.copy()[2:-2, 2:-2, :]
    else:
        raise NotImplementedError
    return x0


def get_A(operator_config):
    """Get forward operator"""
    if operator_config.operator == "cs":
        A = get_cs_A(operator_config)
    else:
        raise NotImplementedError
    return A


def get_prox(args):
    """Get proximal step"""
    if args.prox_config.prox == "lpn":
        model = load_model(args.model_config, args.prox_config.model_path)
        prox = lambda x, lam=1.0, **kwargs: model.apply_numpy(np.asarray(x))

    else:
        raise NotImplementedError

    return prox


def save_results(data_list, out_dir):
    """Save results to disk
    Inputs:
        data_list: list of data to be saved
        out_dir: directory to save results
    """
    os.makedirs(out_dir, exist_ok=True)
    for i, data in enumerate(data_list):
        np.save(os.path.join(out_dir, f"{i}.npy"), data)


def main(args):
    if args.dataset_config.dataset == "mayoct":
        if args.operator_config.operator == "cs":
            main_mayoct_cs(args)
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError

if __name__ == "__main__":
    args = parse_args()
    main(args)
