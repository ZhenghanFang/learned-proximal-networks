"""Solve inverse problems using LPN."""

import argparse
import os
import scico
from scico import functional, linop, loss
from scico.optimize.admm import LinearSubproblemSolver
import numpy as np
import torch
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
from skimage.metrics import structural_similarity as skimage_ssim
from pprint import pp
from PIL import Image


from lpn.utils import get_model, get_imgs, load_config
from lpn.operators.blur import get_blur_A
from lpn.operators.tomo import get_tomo_A
from lpn.operators.cs import get_cs_A
from lpn.utils import scico as nonjit_scico
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
    parser.add_argument("--sigma_blur", type=float)
    args = parser.parse_args()

    # load config
    args.dataset_config = load_config(args.dataset_config_path)
    args.operator_config = load_config(args.operator_config_path)
    args.operator_config.sigma_blur = args.sigma_blur
    args.prox_config = load_config(args.prox_config_path)
    if args.prox_config is not None and args.prox_config.prox == "lpn":
        args.model_config = load_config(
            os.path.join(
                os.path.dirname(args.prox_config.model_path), "model_config.json"
            )
        )
    if args.admm_config_path is not None:
        args.admm_config = load_config(args.admm_config_path)
    if args.data_dir is None:
        args.data_dir = args.out_dir

    print("args:")
    pp(vars(args))

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


def main_celeba_pgd(args):
    A = get_A(args.operator_config)
    x_list = get_imgs(args.dataset_config)
    y_list = [
        np.load(os.path.join(args.out_dir, "meas", "y", f"{i}.npy"))
        for i in range(len(x_list))
    ]

    prox = get_prox(args)

    # parse PGD arguments
    eta = 2.0  # step size
    maxiter = 20  # number of iterations
    x0_type = "adjoint"

    idx_list = np.arange(20)
    metric_list = []
    for i in tqdm(idx_list):
        y = y_list[i]

        x0 = get_x0(y, A, x0_type)

        g = HelperFunctional(prox)

        # PGD solver
        x = x0.copy()
        for step in range(maxiter):
            x = x - eta * A.adj(A(x) - y)
            x = prox(x)
            x = np.clip(x, 0, 1)
            print(
                f"PSNR: {skimage_psnr(x_list[i], x, data_range=1.)}",
                f"SSIM: {skimage_ssim(x_list[i], x, data_range=1., channel_axis=2)}",
            )

        xhat = np.clip(x, 0, 1)
        _save(xhat, os.path.join(args.out_dir, "pgd", args.prox_config.prox, "xhat"), i)
        psnr = skimage_psnr(x_list[i], xhat, data_range=1.0)
        ssim = skimage_ssim(x_list[i], xhat, data_range=1.0, channel_axis=2)
        metric_list.append([psnr, ssim])
    metric_list = np.array(metric_list)
    print(f"PSNR: {metric_list[:, 0].mean():.4f} +- {metric_list[:, 0].std():.4f}")
    print(f"SSIM: {metric_list[:, 1].mean():.4f} +- {metric_list[:, 1].std():.4f}")


def main_celeba(args):
    A = get_A(args.operator_config)
    x_list = get_imgs(args.dataset_config)
    if args.measure:
        y_list = measure(x_list, A, args.sigma_noise, args.seed)
        save_results(x_list, os.path.join(args.out_dir, "meas", "x"))
        save_results(y_list, os.path.join(args.out_dir, "meas", "y"))
        print("Measurements saved.")
        return

    # y_list = [
    #     np.load(os.path.join(args.data_dir, "meas", "y", f"{i}.npy"))
    #     for i in range(len(x_list))
    # ]
    y_list = measure(x_list, A, args.sigma_noise, args.seed)

    prox = get_prox(args)

    # parse ADMM arguments
    rho = args.admm_config.rho  # ADMM penalty parameter
    maxiter = args.admm_config.maxiter  # number of ADMM iterations
    subprob_tol = 1e-5  # tolerance for subproblem
    subprob_maxiter = 30  # maximum iteration for subproblem

    def _callback(opt):
        print(i, x_list[i].shape, opt.x.shape)
        print(
            f"PSNR: {skimage_psnr(x_list[i], np.array(opt.x), data_range=1.)}",
            f"SSIM: {skimage_ssim(x_list[i], np.array(opt.x), data_range=1., channel_axis=2)}",
        )

    idx_list = np.arange(20)
    metric_list = []
    for i in tqdm(idx_list):
        y = y_list[i]

        x0 = get_x0(y, A, args.admm_config.x0)
        x_shape = x0.shape

        f = loss.SquaredL2Loss(y=y, A=A, scale=args.admm_config.scale)

        g = HelperFunctional(prox)

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
            subproblem_solver=scico.optimize.admm.LinearSubproblemSolver(
                cg_kwargs={"tol": subprob_tol, "maxiter": subprob_maxiter}
            ),
            itstat_options={"display": True},
            order=args.admm_config.order,
        )
        xhat = solver.solve(callback=_callback)
        xhat = np.asarray(xhat)
        if args.out_dir is not None:
            _save(xhat, os.path.join(args.out_dir, args.prox_config.prox, "xhat"), i)
            _save(y, os.path.join(args.out_dir, args.prox_config.prox, "y"), i)
            _save(x_list[i], os.path.join(args.out_dir, args.prox_config.prox, "x"), i)
        psnr = skimage_psnr(x_list[i], xhat, data_range=1.0)
        ssim = skimage_ssim(x_list[i], xhat, data_range=1.0, channel_axis=2)
        metric_list.append([psnr, ssim])
    metric_list = np.array(metric_list)
    print(f"PSNR: {metric_list[:, 0].mean():.4f} +- {metric_list[:, 0].std():.4f}")
    print(f"SSIM: {metric_list[:, 1].mean():.4f} +- {metric_list[:, 1].std():.4f}")


def load_model(model_config, model_path):
    """Load LPN model for testing"""
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model


def _save(data, dir, i):
    os.makedirs(dir, exist_ok=True)
    # save npy
    np.save(os.path.join(dir, f"{i}.npy"), data)
    # save png
    data = (data * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(data).save(os.path.join(dir, f"{i}.png"))


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
    if operator_config.operator == "blur":
        sigma_blur = operator_config.sigma_blur
        x_shape = [operator_config.image_size, operator_config.image_size, 3]
        A = get_blur_A(sigma_blur, x_shape)
    else:
        raise NotImplementedError
    return A


def get_prox(args):
    """Get proximal step"""
    if args.prox_config.prox == "bm3d":
        if args.prox_config.lamb == -1:
            args.prox_config.lamb = args.sigma_noise
        print(f"BM3D lambda: {args.prox_config.lamb}")
        prox = (
            functional.ScaledFunctional(functional.BM3D(), scale=args.prox_config.lamb)
        ).prox

    elif args.prox_config.prox == "dncnn":
        prox = functional.DnCNN("17M").prox

    elif args.prox_config.prox == "lpn":
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
    if args.dataset_config.dataset == "celeba":
        if args.solver == "pgd":
            main_celeba_pgd(args)
        else:  # admm
            main_celeba(args)
    else:
        raise NotImplementedError


if __name__ == "__main__":
    args = parse_args()
    main(args)
