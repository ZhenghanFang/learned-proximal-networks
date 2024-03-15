"""Learned proximal networks for input size of 128x128."""


import numpy as np
import torch
from torch import nn


class LPN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden,
        beta,
        alpha,
    ):
        super().__init__()

        self.hidden = hidden
        self.lin = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, bias=True, stride=1, padding=1),  # 128
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=2, padding=1),  # 64
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 64
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=2, padding=1),  # 32
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=1, padding=1),  # 32
                nn.Conv2d(hidden, hidden, 3, bias=False, stride=2, padding=1),  # 16
                nn.Conv2d(hidden, 64, 16, bias=False, stride=1, padding=0),  # 1
                nn.Linear(64, 1),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1),  # 64
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 64
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1),  # 32
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1),  # 32
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1),  # 16
                nn.Conv2d(in_dim, 64, 16, stride=1, padding=0),  # 1
            ]
        )

        self.act = nn.Softplus(beta=beta)
        self.alpha = alpha

    def scalar(self, x):
        bsize = x.shape[0]
        assert x.shape[-1] == x.shape[-2]
        image_size = x.shape[-1]
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [
            image_size,
            image_size // 2,
            image_size // 2,
            image_size // 4,
            image_size // 4,
            image_size // 8,
        ]
        for core, res, sz in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, (sz, sz), mode="bilinear")
            y = self.act(core(y) + res(x_scaled))

        x_scaled = nn.functional.interpolate(x, (size[-1], size[-1]), mode="bilinear")
        y = self.lin[-2](y) + self.res[-1](
            x_scaled
        )  # 1x1 if input is 128x128, 2x2 if input is 136x136
        y = self.act(y)
        # avg pooling
        assert y.shape[2] == y.shape[3] == 1
        y = torch.mean(y, dim=(2, 3))

        y = y.reshape(bsize, 64)
        y = self.lin[-1](y)  # (batch, 1)

        # strongly convex
        y = y + self.alpha * x.reshape(x.shape[0], -1).pow(2).sum(1, keepdim=True)

        # return shape: (batch, 1)
        return y

    def init_weights(self, mean, std):
        print("init weights")
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.normal_(mean, std).exp_()

    # this clips the weights to be non-negative to preserve convexity
    def wclip(self):
        with torch.no_grad():
            for core in self.lin[1:]:
                core.weight.data.clamp_(0)

    def forward(self, x):
        with torch.enable_grad():
            if not x.requires_grad:
                x.requires_grad = True
            x_ = x
            y = self.scalar(x_)
            grad = torch.autograd.grad(
                y.sum(), x_, retain_graph=True, create_graph=True
            )[0]

        return grad

    def apply_numpy(self, x):
        """Apply LPN to a numpy image.
        Inputs:
            x: image, shape (H, W) or (H, W, C)
        Outputs:
            shape (H, W) or (H, W, C)
        """
        assert x.shape[:2] == (128, 128)  # image size must be 128x128
        # get device of model
        device = next(self.parameters()).device
        x_dim = len(x.shape)
        if x_dim == 2:
            x = x[:, :, np.newaxis]
        x = np.transpose(x, (2, 0, 1))
        x = torch.tensor(x).unsqueeze(0).to(device)
        with torch.no_grad():
            x = self(x)
        x = x[0].detach().cpu().numpy()
        x = np.transpose(x, (1, 2, 0))
        if x_dim == 2:
            x = np.squeeze(x, 2)
        return x
