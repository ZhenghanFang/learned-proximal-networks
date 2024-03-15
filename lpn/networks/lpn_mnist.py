"""Learned proximal networks for input size of 28x28 (e.g. MNIST)."""

import torch
from torch import nn


def get_padding(kernel_size, dilation):
    return (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2


class LPN(nn.Module):
    def __init__(
        self,
        in_dim,
        hidden,
        layers=1,
        beta=1,
        kernel_size=3,
        stride=1,
        dilation=1,
        alpha=0.0,
    ):
        super(LPN, self).__init__()

        # conv kernel size
        if type(kernel_size) == int:
            kernel_size = [kernel_size] * layers
        # conv dilation
        if type(dilation) == int:
            dilation = [dilation] * layers
        # conv stride
        if type(stride) == int:
            stride = [stride] * layers

        self.hidden = hidden
        self.lin = nn.ModuleList(
            [
                nn.Conv2d(
                    in_dim, hidden, 3, bias=False, stride=1, padding=1, dilation=1
                ),  # 28
                nn.Conv2d(
                    hidden, hidden, 3, bias=False, stride=2, padding=1, dilation=1
                ),  # 14
                nn.Conv2d(
                    hidden, hidden, 3, bias=False, stride=1, padding=1, dilation=1
                ),  # 14
                nn.Conv2d(
                    hidden, hidden, 3, bias=False, stride=2, padding=1, dilation=1
                ),  # 7
                nn.Linear(hidden * 7 * 7, 64, bias=False),
                nn.Linear(64, 1, bias=False),
            ]
        )

        self.res = nn.ModuleList(
            [
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1, dilation=1),  # 14
                nn.Conv2d(in_dim, hidden, 3, stride=1, padding=1, dilation=1),  # 14
                nn.Conv2d(in_dim, hidden, 3, stride=2, padding=1, dilation=1),  # 7
                nn.Linear(7 * 7 * in_dim, 64),
            ]
        )

        self.act = nn.Softplus(beta=beta)
        self.alpha = alpha

    def scalar(self, x):
        y = x.clone()
        y = self.act(self.lin[0](y))
        size = [28, 14, 14, 7]
        for core, res, sz in zip(self.lin[1:-2], self.res[:-1], size[:-1]):
            x_scaled = nn.functional.interpolate(x, (sz, sz), mode="bilinear")
            y = self.act(core(y) + res(x_scaled))

        y = y.reshape(y.shape[0], -1)
        x_scaled = nn.functional.interpolate(
            x, (size[-1], size[-1]), mode="bilinear"
        ).reshape(x.shape[0], -1)
        y = self.lin[-2](y) + self.res[-1](x_scaled)
        y = self.act(y)

        y = self.lin[-1](y)
        # return shape: (batch, 1)

        y = y + self.alpha * x.pow(2).sum(dim=(1, 2, 3)).unsqueeze(1)

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
