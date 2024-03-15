import importlib
import numpy as np
import torch
from torch import nn
import json
from omegaconf import OmegaConf
import os

from lpn.datasets.mnist import MNISTDataset
from lpn.datasets.celeba import CelebADataset
from lpn.datasets.mayoct import MayoCTDataset


def get_model(model_config):
    """Load model from config file.
    Parameters:
        model_config (OmegaConf): Model config.
    """
    model = importlib.import_module("lpn.networks." + model_config.model).LPN(
        **model_config.params
    )
    model.init_weights(-10, 0.1)
    return model


def _load_model_helper(model_config, model_path):
    """Helper for loading LPN model for testing"""
    model = get_model(model_config)
    model.load_state_dict(torch.load(model_path)["model_state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval().to(device)
    return model


def load_model(model_path):
    """Load LPN model for testing"""
    model_config = load_config(
        os.path.join(os.path.dirname(model_path), "model_config.json")
    )
    return _load_model_helper(model_config, model_path)


def load_dataset(dataset_config, split):
    """Load dataset from config file.
    Parameters:
        dataset_config (OmegaConf): Dataset config.
        split (str): Split of dataset to load.
    """
    dataset = importlib.import_module(
        "lpn.datasets." + dataset_config.dataset
    ).LPNDataset(**dataset_config.params, split=split)
    print("dataset: ", dataset_config.dataset)
    return dataset


def get_loss_hparams_and_lr(args, global_step):
    """Get loss hyperparameters and learning rate based on training schedule.
    Parameters:
        args (argparse.Namespace): Arguments from command line.
        global_step (int): Current training step.
    """
    if global_step < args.num_steps_pretrain:
        loss_hparams, lr = {"type": "l1"}, args.pretrain_lr
    else:
        num_steps = args.num_steps - args.num_steps_pretrain
        step = global_step - args.num_steps_pretrain

        def _get_loss_hparams_and_lr(num_steps, step):
            num_steps_per_stage = num_steps // args.num_stages
            stage = step // num_steps_per_stage
            if stage >= args.num_stages:
                stage = args.num_stages - 1
            loss_hparams = {
                "type": "prox_matching",  # proximal matching
                "sigma": args.sigma_min * (2 ** (args.num_stages - 1 - stage)),
            }
            lr = args.lr
            return loss_hparams, lr

        loss_hparams, lr = _get_loss_hparams_and_lr(num_steps, step)

    return loss_hparams, lr


def get_loss(loss_hparams):
    """Get loss function from hyperparameters.
    Parameters:
        loss_hparams (dict): Hyperparameters for loss function.
    """
    if loss_hparams["type"] == "l1":
        return nn.L1Loss()
    elif loss_hparams["type"] == "prox_matching":
        return ExpDiracSrgt(sigma=loss_hparams["sigma"])
    else:
        raise NotImplementedError


# surrogate L0 loss: -exp(-(x/sigma)^2) + 1
def exp_func(x, sigma):
    return -torch.exp(-((x / sigma) ** 2)) + 1


class ExpDiracSrgt(nn.Module):
    def __init__(self, sigma):
        super().__init__()
        self.sigma = sigma

    def forward(self, input, target):
        """
        input, target: batch, *
        """
        bsize = input.shape[0]
        dist = (input - target).pow(2).reshape(bsize, -1).sum(1).sqrt()
        return exp_func(dist, self.sigma).mean()


def center_crop(img, shape):
    """Center crop image to desired shape.
    Args:
        img: image to be cropped, (h, w, c), numpy array
        shape: desired shape, (h, w)
    Returns:
        img_crop: cropped image, (h, w, c), numpy array
    """
    h, w = img.shape[:2]
    h1, w1 = shape
    assert (h - h1) % 2 == 0 and (w - w1) % 2 == 0
    h_start = (h - h1) // 2
    w_start = (w - w1) // 2
    img_crop = img[h_start : h_start + h1, w_start : w_start + w1, ...]
    return img_crop


def get_imgs(dataset_config):
    """Get images"""
    if dataset_config.dataset == "mnist":
        x_list = get_mnist(dataset_config)
    elif dataset_config.dataset == "celeba":
        x_list = get_celeba(dataset_config)
    elif dataset_config.dataset == "mayoct":
        x_list = get_mayoct(dataset_config)
    else:
        raise NotImplementedError
    return x_list


def get_mnist(config):
    """Get MNIST images"""
    dataset = MNISTDataset(root="data/mnist", split=config.split)
    x_list = []
    for idx in range(config.start_idx, config.start_idx + config.num_imgs):
        img = dataset[idx]["image"]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # (c, h, w) -> (h, w, c)
        if config.squeeze:
            img = np.squeeze(img, 2)
        x_list.append(img)
    return x_list


def get_celeba(config):
    """Get CelebA images"""
    dataset = CelebADataset(root=config.root, split="valid", image_size=128)
    x_list = []
    for idx in range(config.start_idx, config.start_idx + config.num_imgs):
        img = dataset[idx]["image"]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))
        x_list.append(img)
    return x_list


def get_mayoct(config):
    """Get MayoCT images"""
    dataset = MayoCTDataset(root=config.root, split=config.split)
    x_list = []
    for idx in range(config.start_idx, config.start_idx + config.num_imgs):
        img = dataset[idx]["image"]
        img = img.numpy()
        img = np.transpose(img, (1, 2, 0))  # (c, h, w) -> (h, w, c)
        if config.squeeze:
            img = np.squeeze(img, 2)
        x_list.append(img)
    return x_list


def load_config(config_path):
    if config_path is None:
        return None
    with open(config_path, "r") as f:
        config = json.load(f)
    config = OmegaConf.create(config)
    return config
