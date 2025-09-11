import torch
from tqdm import tqdm


def invert_ls(x, model, max_iter=1000, verbose=True):
    """Invert the LPN model at x by least squares min_y||f_theta(y) - x||_2^2.
    Inputs:
        x: (n, *), numpy.ndarray, n points
        model: LPN model.
    Outputs:
        y: (n, *), numpy.ndarray, n points, the inverse
    """
    device = next(model.parameters()).device
    x = torch.tensor(x).float().to(device)
    y = torch.zeros(x.shape).float().to(device)
    y.requires_grad_(True)

    optimizer = torch.optim.Adam([y], lr=1e-2)

    for i in tqdm(range(max_iter), disable=not verbose):
        optimizer.zero_grad()
        loss = (model(y) - x).pow(2).mean()
        loss.backward()
        optimizer.step()
        if i % 100 == 0:
            print(f"iter {i}: mse {loss.item()}")
    print("final mse", loss.item())

    y = y.detach().cpu().numpy()
    print("max, min:", y.max(), y.min())

    return y