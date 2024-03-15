"""Module for inverting LPN using convex optimization with conjugate gradient."""
import numpy as np
from scipy.optimize import fmin_cg
import torch

from .utils import prox


def invert_cvx_cg_single(x, model):
    """Invert the learned proximal operator at x by convex optimization with conjugate gradient.
    Inputs:
        x: (*), a point, numpy array
        model: LPN model.
    Outputs:
        y: (*), a point, numpy array

    The shape of x should match the input shape of the model, i.e., (c, w, h)
    """

    z = x.copy()

    # x0 = np.zeros(z.shape).flatten()
    x0 = z.copy().flatten()
    args = (model, z)
    x_list = []
    callback = lambda x: x_list.append(x)
    res = fmin_cg(
        f, x0, fprime=gradf, args=args, full_output=True, disp=0, callback=callback
    )
    # print(
    #     f"fopt: {res[1]}, func_calls: {res[2]}, grad_calls: {res[3]}, warnflag: {res[4]}"
    # )
    return x_list[-1].reshape(z.shape)


def invert_cvx_cg(x, model):
    """Invert the learned proximal operator at x by convex optimization with conjugate gradient.
    Inputs:
        x: (n, *), a vector of n points, numpy array
        model: LPN model.
    Outputs:
        y: (n, *), a vector of n points, numpy array

    The shape of x should match the input shape of the model, i.e., (n, c, w, h)
    """
    y = np.zeros(x.shape)
    for i in range(x.shape[0]):
        y[i] = invert_cvx_cg_single(x[i], model)
    print("final mse: ", np.mean((prox(y, model) - x) ** 2))
    # print("max, min:", y.max(), y.min())
    return y


# invert
def torch_f(x, net, z):
    """
    Compute the value of the objective function:
    psi(x) - <z, x>

    x: torch tensor, shape (channel, img_size, img_size)
    net: ICNN
    z: torch tensor, shape (channel, img_size, img_size)
    Return: torch tensor, shape (1,)
    """
    v = net.scalar(x.unsqueeze(0)).squeeze() - torch.sum(z.reshape(-1) * x.reshape(-1))
    return v


def f(x, *args):
    """
    x: numpy array, shape (channel * img_size * img_size,)
    args: (net, z)
        net: LPN model
        z: numpy array, shape (channel, img_size, img_size)
    Return: numpy array, shape (1,)
    """
    net, z = args
    x = torch.tensor(x).view(z.shape).float()
    z = torch.tensor(z).float()
    device = next(net.parameters()).device
    x = x.to(device)
    z = z.to(device)
    v = torch_f(x, net, z)
    v = v.cpu().detach().numpy()
    return v


def gradf(x, *args):
    """
    x: numpy array, shape (channel * img_size * img_size,)
    args: (net, z)
        net: LPN model
        z: numpy array, shape (channel, img_size, img_size)
    Return: numpy array, shape (channel * img_size * img_size,)
    """
    net, z = args
    x = torch.tensor(x).view(z.shape).float()
    z = torch.tensor(z).float()
    device = next(net.parameters()).device
    x = x.to(device)
    z = z.to(device)
    x.requires_grad_(True)
    v = torch_f(x, net, z)
    v.backward()
    g = x.grad.cpu().numpy().flatten()
    return g
