"""Compressed sensing operator."""


from argparse import Namespace
import numpy as np
import scico
from scico import linop
import torch
import os


def get_cs_A(cfg):
    """Get compressed sensing operator A."""
    if cfg.init:
        return CSOperatorScico(cfg.img_size, cfg.n_measurements)
    else:
        return CSOperatorScico.from_file(cfg.operator_path)


class CSOperatorScico(linop.LinearOperator):
    """Scico wrapper for compressed sensing operator."""

    def __init__(self, img_size, n_measurements):
        hparams = Namespace()
        hparams.n_measurements = n_measurements
        hparams.img_size = img_size
        hparams.n_input = np.prod(img_size)
        hparams.A = np.random.randn(hparams.n_input)
        hparams.sign_pattern = np.float32(
            (np.random.rand(hparams.n_input) < 0.5) * 2 - 1.0
        )
        hparams.indices = np.random.choice(
            hparams.n_input, hparams.n_measurements, replace=False
        )

        self.hparams = hparams
        self.op = CSOperatorTorch(hparams)

        self._init_scico()

    def _init_scico(self):
        fwd_op_numpy = lambda x: self.op(torch.from_numpy(np.array(x))).cpu().numpy()
        adjoint_op_numpy = (
            lambda x: self.op.adjoint(torch.from_numpy(np.array(x))).cpu().numpy()
        )

        super().__init__(
            input_shape=(self.hparams.img_size),
            output_shape=(self.hparams.n_measurements,),
            eval_fn=fwd_op_numpy,
            adj_fn=adjoint_op_numpy,
            jit=False,
        )

    def save_operator(self, fn):
        os.makedirs(os.path.dirname(fn), exist_ok=True)
        np.save(fn, self.hparams)

    def load_operator(self, fn):
        hparams = np.load(fn, allow_pickle=True).item()
        self.hparams = hparams
        self.op = CSOperatorTorch(hparams)
        self._init_scico()

    @classmethod
    def from_file(cls, fn):
        hparams = np.load(fn, allow_pickle=True).item()
        ins = cls(hparams.img_size, hparams.n_measurements)
        ins.load_operator(fn)
        return ins

    def get_norm(self):
        return self.op.norm()

    def get_ls(self, y):
        return self.op.ls(torch.tensor(y)).numpy()


class CSOperatorTorch(torch.nn.Module):
    """Torch wrapper for compressed sensing operator."""

    def __init__(self, hparams):
        """
        Inputs:
            hparams: Namespace
                A: (n_input)
                sign_pattern: (n_input)
                indices: (n_measurements)
                n_measurements: int
                n_input: int
        """
        super().__init__()
        self.n_measurements = hparams.n_measurements
        self.n_input = hparams.n_input

        # to torch
        A = torch.from_numpy(hparams.A).float()
        sign_pattern = torch.from_numpy(hparams.sign_pattern).float()
        indices = torch.from_numpy(hparams.indices).long()
        self.indices = indices

        # register buffers
        self.register_buffer("A", A)
        self.register_buffer("sign_pattern", sign_pattern)

        # input output size
        self.input_size = (int(np.sqrt(self.n_input)), int(np.sqrt(self.n_input)))
        self.output_size = (self.n_measurements,)

    def forward(self, x):
        """Inputs:
            x: (sqrt(n_input), sqrt(n_input)), torch tensor
        Outputs:
            y: (n_measurements)
        """
        x = x.reshape((self.n_input))
        z = _ifft(_fft(x * self.sign_pattern) * _fft(self.A))
        y = z[self.indices]
        y = torch.real(y)
        return y

    def adjoint(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        z = torch.zeros(self.n_input, dtype=torch.float)
        z[self.indices] = y
        x = _ifft(_fft(z) * torch.conj(_fft(self.A)))
        x = x * self.sign_pattern
        x = x.reshape(self.input_size)
        x = torch.real(x)
        return x

    def adjoint_vjp(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        inputs = torch.zeros(self.input_size, dtype=torch.complex64)
        _, vjp = torch.autograd.functional.vjp(self.forward, inputs, y)
        vjp = torch.real(vjp)
        return vjp

    def pinv(self, y):
        """Inputs:
            y: (n_measurements), torch tensor
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        z = torch.zeros(self.n_input)
        z[self.indices] = y
        x = _ifft(_fft(z) / _fft(self.A))
        x = x / self.sign_pattern
        x = x.reshape(self.input_size)
        x = torch.real(x)
        return x

    def ls(self, y, d=1e-6, verbose=False):
        """
        Least square solution
        Inputs:
            y: (n_measurements), torch tensor
            d: damping factor, see https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html#scipy.sparse.linalg.lsqr
        Outputs:
            x: (sqrt(n_input), sqrt(n_input))
        """
        # use gradient descent
        x = torch.zeros(self.input_size, dtype=torch.float)
        eta = 1e-1
        rel_change = []
        for _ in range(100):
            x_old = x.clone()
            x = x - eta * (self.adjoint(self.forward(x) - y) - x * d)
            rel_change.append((torch.norm(x - x_old) / torch.norm(x)).item())
            if rel_change[-1] < 1e-3:
                if verbose:
                    print("ls rel change converged!")
                break
        else:
            if verbose:
                print("Warning: ls rel change not converged!")
        return x

    def gram(self, x):
        """
        Inputs:
            x: (sqrt(n_input), sqrt(n_input)), torch tensor
        Outputs:
            y: (n_measurements)
        """
        return self.adjoint(self.forward(x))

    def norm(self, verbose=False):
        """
        Find operator norm by power method.
        (the largest singular value of the measurement matrix)
        """
        x = torch.randn(self.input_size, dtype=torch.float)
        for i in range(1000):
            x = self.gram(x)
            x = x / torch.norm(x)

            # Rayleigh quotient (largest eigenvalue of the gram matrix)
            R = torch.sum(self.gram(x) * x) / (torch.norm(x) ** 2)

            # operator norm (square root of largest eigenvalue)
            n = torch.sqrt(R)

            if i > 0 and torch.abs(n - n_old) < 1e-2:
                if verbose:
                    print("norm converged!")
                break

            n_old = n
        else:
            if verbose:
                print("Warning: norm not converged!")
        return n.item()


def _fft(x):
    return torch.fft.fft(x, norm="ortho")


def _ifft(x):
    return torch.fft.ifft(x, norm="ortho")
