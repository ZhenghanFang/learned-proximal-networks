import torch


def prox(x, model):
    """Evaluate the learned proximal operator at x.
    Inputs:
        x: (n, *), n points, numpy array
        model: LPN model.
    Outputs:
        y: (n, *), n points, numpy array

    The shape of x should match the input shape of the model.
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    return model(x).detach().cpu().numpy()


def cvx(x, model):
    """Evaluate the learned convex function at x.
    Inputs:
        x: (n, *), n points, numpy array
        model: LPN model.
    Outputs:
        y: (n, ), a vector of n values, numpy array

    The shape of x should match the input shape of the model.
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    return model.scalar(x).squeeze(1).detach().cpu().numpy()
