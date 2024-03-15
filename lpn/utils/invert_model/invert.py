from .invert_cvx_cg import invert_cvx_cg
from .invert_cvx_gd import invert_cvx_gd
from .invert_ls import invert_ls


def invert(x, model, inv_alg):
    """Invert the LPN model at x.
    Inputs:
        x: (n, *), numpy.ndarray, n points
        model: LPN model.
        inv_alg: Inversion algorithm, choose from ['ls', 'cvx_cg', 'cvx_gd']
    Outputs:
        y: (n, *), numpy.ndarray, n points, the inverse

    Note: The shape of x should match the input shape of the model.
    """
    if inv_alg == "ls":
        return invert_ls(x, model)
    elif inv_alg == "cvx_cg":
        return invert_cvx_cg(x, model)
    elif inv_alg == "cvx_gd":
        return invert_cvx_gd(x, model)
    else:
        raise ValueError("Unknown inversion algorithm:", inv_alg)
