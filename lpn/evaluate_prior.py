"""Evaluate learned prior of LPN."""
import torch
import numpy as np
import random
import os
import argparse
from tqdm import tqdm
import pandas as pd

from utils import get_model, get_imgs, load_config
from utils.perturb import perturb
from utils.prior import evaluate_prior


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="The config of the LPN model.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="output directory. If None, save to {model directory}/prior.",
    )
    parser.add_argument(
        "--perturb_config_path", type=str, default=None, help="perturb config"
    )
    parser.add_argument(
        "--dataset_config_path", type=str, default=None, help="dataset config"
    )
    parser.add_argument("--model_path", type=str, default=None, help="model path")
    parser.add_argument(
        "--inv_alg",
        type=str,
        default="cvx_cg",
        help="Inversion algorithm. Choose from ['ls', 'cvx_cg', 'cvx_gd']",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # load config
    args.model_config = load_config(args.model_config_path)
    args.perturb_config = load_config(args.perturb_config_path)
    args.dataset_config = load_config(args.dataset_config_path)

    # pretty print args

    return args


def main(args):
    INV_ALG = args.inv_alg
    # set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # define perturbation
    perturb_mode = args.perturb_config.perturb_mode
    sigma_list = args.perturb_config.sigma_list

    # load model
    model = get_model(args.model_config)
    model.load_state_dict(torch.load(args.model_path)["model_state_dict"])
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # get data
    img_list = get_imgs(args.dataset_config)
    imgs = np.array(img_list)
    imgs = np.transpose(imgs, (0, 3, 1, 2))  # (n, c, h, w)

    ndim = np.prod(imgs.shape[1:])

    if perturb_mode == "convex":
        # get images for convex combination
        imgs2 = imgs[::-1]
        # assert even number of images, so that no pair contains the same image
        assert imgs.shape[0] % 2 == 0

    # compute prior and inverse
    p_list = []
    x_list = []
    y_list = []
    fy_list = []
    for i, sigma in enumerate(tqdm(sigma_list)):
        # perturb
        perturb_kw = {"sigma": sigma}
        if perturb_mode == "convex":
            perturb_kw["z"] = imgs2
        x = perturb(imgs, perturb_mode, perturb_kw)

        # evaluate_prior
        res = evaluate_prior(x, model, INV_ALG)
        p = res["p"]
        y = res["y"]
        fy = res["fy"]

        # print("x", x.shape)
        # print("y", y.shape)
        # print("p", p.shape)
        # print(p)
        p_array = p
        x_array = x.reshape(len(imgs), ndim)
        y_array = y.reshape(len(imgs), ndim)
        fy_array = fy.reshape(len(imgs), ndim)

        save_dir = f"{args.out_dir}/{INV_ALG}/{perturb_mode}/sigma={sigma}"
        os.makedirs(save_dir, exist_ok=True)

        np.save(os.path.join(save_dir, "prior.npy"), p_array)
        np.save(os.path.join(save_dir, "x.npy"), x_array)
        np.save(os.path.join(save_dir, "y.npy"), y_array)
        np.save(os.path.join(save_dir, "fy.npy"), fy_array)

        p_list.append(p_array)
        x_list.append(x_array)
        y_list.append(y_array)
        fy_list.append(fy_array)

    df = pd.DataFrame(np.array(p_list).T, columns=sigma_list)
    save_dir = f"{args.out_dir}/{INV_ALG}/{perturb_mode}/all"
    os.makedirs(save_dir, exist_ok=True)
    df.to_csv(f"{save_dir}/prior.csv", index=False)
    np.save(f"{save_dir}/x.npy", np.stack(x_list, axis=1))  # (n_img, n_sigma, n_dim)
    np.save(f"{save_dir}/y.npy", np.stack(y_list, axis=1))
    np.save(f"{save_dir}/fy.npy", np.stack(fy_list, axis=1))



if __name__ == "__main__":
    args = parse_args()
    main(args)
