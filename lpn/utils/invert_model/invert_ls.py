import torch


def invert_ls(x, model):
    """Invert the LPN model at x by least squares min_y\|f_\theta(y) - x\|_2^2.
    Inputs:
        x: (n, *), numpy.ndarray, n points
        model: LPN model.
    Outputs:
        y: (n, *), numpy.ndarray, n points, the inverse
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    y = torch.zeros(x.shape).float().to(device)

    optimizer = torch.optim.Adam([y], lr=1e-2)

    for i in range(1000):
        optimizer.zero_grad()
        loss = (model(y) - x).pow(2).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print("mse", loss.item())
    print("final mse", loss.item())

    y = y.detach().cpu().numpy()
    print("max, min:", y.max(), y.min())

    return y
