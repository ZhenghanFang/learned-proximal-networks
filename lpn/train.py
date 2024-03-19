import argparse
from pprint import pp
import torch
from torch import nn
from tqdm import tqdm
import numpy as np
import json
import os
from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter

from utils import load_dataset
from utils import get_model
from utils import get_loss_hparams_and_lr, get_loss
from utils import trainer
from utils import utils


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_dir",
        type=str,
        default="./experiments/celeba",
        help="The experiment directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--dataset_config_path",
        type=str,
        default=None,
        help=(
            "The name of the Dataset to train on. It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that HF Datasets can understand."
        ),
    )
    parser.add_argument(
        "--model_config_path",
        type=str,
        default=None,
        help="The config of the model to train.",
    )
    parser.add_argument(
        "--image_size", type=int, default=128, help="Input image (patch) size of LPN."
    )
    parser.add_argument(
        "--num_channels", type=int, default=3, help="Number of image channels."
    )
    parser.add_argument(
        "--sigma_min",
        type=float,
        default=None,
        help="Minimum gamma in proximal matching loss.",
    )
    parser.add_argument("--sigma_noise", type=float, default=0.1, help="Noise level.")
    parser.add_argument(
        "--sigma_noise_min", type=float, default=0.01, help="Min noise level."
    )
    parser.add_argument(
        "--sigma_noise_max", type=float, default=0.1, help="Max noise level."
    )
    parser.add_argument(
        "--random_noise_level", action="store_true", help="Random noise level."
    )
    parser.add_argument("--train_batch_size", type=int, default=64)
    parser.add_argument("--dataloader_num_workers", type=int, default=8)
    parser.add_argument(
        "--num_steps", type=int, default=40000, help="Number of training steps."
    )
    parser.add_argument(
        "--num_steps_pretrain",
        type=int,
        default=20000,
        help="Number of ell_1 pretrain steps.",
    )
    parser.add_argument(
        "--pretrain_lr",
        type=float,
        default=1e-3,
        help="ell_1 pretrain learning rate.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate for proximal matching loss.",
    )
    parser.add_argument(
        "--save_every_n_steps", type=int, default=1000, help="Save model every n steps."
    )
    parser.add_argument(
        "--validate_every_n_steps",
        type=int,
        default=0,
        help="Validate model every n steps. Set to 0 to disable validation.",
    )
    parser.add_argument(
        "--num_stages",
        type=int,
        default=4,
        help="Number of stages in proximal matching gamma ramp down.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Strong convexity coefficient for LPN.",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to a checkpoint to resume from (if any).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="adam",
        choices=["adam", "sgd"],
        help="Optimizer to use.",
    )
    args = parser.parse_args()

    if args.sigma_min is None:
        args.sigma_min = (
            0.08
            * np.sqrt(args.image_size * args.image_size * args.num_channels)
            # * (args.sigma_noise / 0.1)
        )

    with open(args.model_config_path, "r") as f:
        model_config = json.load(f)
    args.model_config = OmegaConf.create(model_config)

    with open(args.dataset_config_path, "r") as f:
        dataset_config = json.load(f)
    args.dataset_config = OmegaConf.create(dataset_config)

    if args.alpha is not None:
        args.model_config.params.alpha = args.alpha

    if args.random_noise_level:
        args.sigma_noise = [args.sigma_noise_min, args.sigma_noise_max]

    print("args:")
    pp(vars(args))

    return args


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.exp_dir, exist_ok=True)

    # Initialize the model
    model = get_model(args.model_config).to(device)
    if args.resume_from:
        print(f"Resuming from {args.resume_from}")
        # checkpoint = torch.load(args.resume_from)
        # model.load_state_dict(checkpoint["model_state_dict"])
        model = utils.load_model(args.resume_from)

    # Initialize the optimizer
    if args.optimizer == "adam":
        optimizer = torch.optim.Adam(model.parameters())
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    # tensorboard
    writer = SummaryWriter(log_dir=f"{args.exp_dir}/tb")

    # Get the dataset
    train_dataset = load_dataset(args.dataset_config, "train")

    # Get the dataloader
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    if args.validate_every_n_steps > 0:
        valid_dataset = load_dataset(args.dataset_config, "valid")
        valid_dataloader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
        )
        validator = trainer.Validator(valid_dataloader, writer, args.sigma_noise)

    global_step = 0
    progress_bar = tqdm(total=args.num_steps)
    progress_bar.set_description(f"Train")
    while True:
        for step, batch in enumerate(train_dataloader):
            if args.validate_every_n_steps > 0 and global_step % args.validate_every_n_steps == 0:
                validator.validate(model, global_step)
                
            model.train()
            # get loss hyperparameters and learning rate
            loss_hparams, lr = get_loss_hparams_and_lr(args, global_step)

            # get loss
            loss_func = get_loss(loss_hparams)
            # set learning rate
            for g in optimizer.param_groups:
                g["lr"] = lr

            # Train step
            result = train_step(
                model, optimizer, batch, loss_func, args.sigma_noise, device
            )
            loss = result["loss"]

            logs = {
                "loss": loss.detach().item(),
                "set": loss_hparams,
                "lr": lr,
            }
            progress_bar.update(1)
            progress_bar.set_postfix(**logs)

            

            if global_step % args.save_every_n_steps == 0:
                fn = os.path.join(args.exp_dir, "model.pt")
                save_checkpoint(args, global_step, model, optimizer, loss, fn)

            global_step += 1
            if global_step >= args.num_steps:
                break

        if global_step >= args.num_steps:
            break

    progress_bar.close()
    fn = os.path.join(args.exp_dir, "model.pt")
    save_checkpoint(args, global_step, model, optimizer, loss, fn)


def save_checkpoint(args, global_step, model, optimizer, loss, filename):
    torch.save(
        {
            "iteration": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": loss.detach().item(),
        },
        filename,
    )

    # save model config
    fn = os.path.join(os.path.dirname(filename), "model_config.json")
    with open(fn, "w") as f:
        json.dump(OmegaConf.to_container(args.model_config), f, indent=4)


def train_step(model, optimizer, batch, loss_func, sigma_noise, device):
    clean_images = batch["image"].to(device)
    noise = torch.randn_like(clean_images)
    if type(sigma_noise) == list:
        # uniform random noise level
        sigma_noise = (
            torch.rand(1).to(noise.device) * (sigma_noise[1] - sigma_noise[0])
            + sigma_noise[0]
        )
    noisy_images = clean_images + sigma_noise * noise
    out = model(noisy_images)

    loss = loss_func(out, clean_images)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model.wclip()  # clip weights to non-negative values to ensure convexity

    result = {"loss": loss}
    return result


if __name__ == "__main__":
    args = parse_args()
    main(args)
