###### scico imports ######
import scico
import scico.numpy as snp
from scico import metric, plot
from scico import functional
from scico import linop, loss
from scico.linop import Diagonal, Identity, operator_norm
from scico.linop.radon_astra import TomographicProjector
from scico.optimize import PGM, AcceleratedPGM
from scico.optimize.pgm import PGMStepSize
from scico.optimize.admm import ADMM, LinearSubproblemSolver, SubproblemSolver
from scico.util import device_info



import scico.optimize.admm as soa
from scico.solver import cg as scico_cg
from scico.loss import SquaredL2Loss
from scico.linop import LinearOperator
from functools import reduce
from scico.numpy.util import ensure_on_device, is_real_dtype
###### LinearSubproblemSolver without jit ######
class LinearSubproblemSolver(SubproblemSolver):
    r"""Solver for quadratic functionals.

    Solver for the case in which :code:`f` is a quadratic function of
    :math:`\mb{x}`. It is a specialization of :class:`.SubproblemSolver`
    for the case where :code:`f` is an :math:`\ell_2` or weighted
    :math:`\ell_2` norm, and :code:`f.A` is a linear operator, so that
    the subproblem involves solving a linear equation. This requires that
    :code:`f.functional` be an instance of :class:`.SquaredL2Loss` and
    for the forward operator :code:`f.A` to be an instance of
    :class:`.LinearOperator`.

    The :math:`\mb{x}`-update step is

    ..  math::

        \mb{x}^{(k+1)} = \argmin_{\mb{x}} \; \frac{1}{2}
        \norm{\mb{y} - A \mb{x}}_W^2 + \sum_i \frac{\rho_i}{2}
        \norm{\mb{z}^{(k)}_i - \mb{u}^{(k)}_i - C_i \mb{x}}_2^2 \;,

    where :math:`W` a weighting :class:`.Diagonal` operator
    or an :class:`.Identity` operator (i.e., no weighting).
    This update step reduces to the solution of the linear system

    ..  math::

        \left(A^H W A + \sum_{i=1}^N \rho_i C_i^H C_i \right)
        \mb{x}^{(k+1)} = \;
        A^H W \mb{y} + \sum_{i=1}^N \rho_i C_i^H ( \mb{z}^{(k)}_i -
        \mb{u}^{(k)}_i) \;.


    Attributes:
        admm (:class:`.ADMM`): ADMM solver object to which the solver is
            attached.
        cg_kwargs (dict): Dictionary of arguments for CG solver.
        cg (func): CG solver function (:func:`scico.solver.cg` or
            :func:`jax.scipy.sparse.linalg.cg`) lhs (type): Function
            implementing the linear operator needed for the
            :math:`\mb{x}` update step.
    """

    def __init__(self, cg_kwargs=None, cg_function= "scico"):
        """Initialize a :class:`LinearSubproblemSolver` object.

        Args:
            cg_kwargs: Dictionary of arguments for CG solver. See
                documentation for :func:`scico.solver.cg` or
                :func:`jax.scipy.sparse.linalg.cg`,
                including how to specify a preconditioner.
                Default values are the same as those of
                :func:`scico.solver.cg`, except for
                `"tol": 1e-4` and `"maxiter": 100`.
            cg_function: String indicating which CG implementation to
                use. One of "jax" or "scico"; default "scico". If
                "scico", uses :func:`scico.solver.cg`. If "jax", uses
                :func:`jax.scipy.sparse.linalg.cg`. The "jax" option is
                slower on small-scale problems or problems involving
                external functions, but can be differentiated through.
                The "scico" option is faster on small-scale problems, but
                slower on large-scale problems where the forward
                operator is written entirely in jax.
        """

        default_cg_kwargs = {"tol": 1e-4, "maxiter": 100}
        if cg_kwargs:
            default_cg_kwargs.update(cg_kwargs)
        self.cg_kwargs = default_cg_kwargs
        self.cg_function = cg_function
        if cg_function == "scico":
            self.cg = scico_cg
        elif cg_function == "jax":
            self.cg = jax_cg
        else:
            raise ValueError(
                f"Parameter cg_function must be one of 'jax', 'scico'; got {cg_function}."
            )
        self.info = None

    def internal_init(self, admm: soa.ADMM):
        if admm.f is not None:
            if not isinstance(admm.f, SquaredL2Loss):
                raise TypeError(
                    "LinearSubproblemSolver requires f to be a scico.loss.SquaredL2Loss; "
                    f"got {type(admm.f)}."
                )
            if not isinstance(admm.f.A, LinearOperator):
                raise TypeError(
                    "LinearSubproblemSolver requires f.A to be a scico.linop.LinearOperator; "
                    f"got {type(admm.f.A)}."
                )

        super().internal_init(admm)

        # Set lhs_op =  \sum_i rho_i * Ci.H @ Ci
        # Use reduce as the initialization of this sum is messy otherwise
        lhs_op = reduce(
            lambda a, b: a + b, [rhoi * Ci.gram_op for rhoi, Ci in zip(admm.rho_list, admm.C_list)]
        )
        if admm.f is not None:
            # hessian = A.T @ W @ A; W may be identity
            lhs_op += admm.f.hessian

        # lhs_op.jit()
        self.lhs_op = lhs_op

    def compute_rhs(self):
        r"""Compute the right hand side of the linear equation to be solved.

        Compute

        .. math::

            A^H W \mb{y} + \sum_{i=1}^N \rho_i C_i^H ( \mb{z}^{(k)}_i -
            \mb{u}^{(k)}_i) \;.

        Returns:
            Computed solution.
        """

        C0 = self.admm.C_list[0]
        rhs = snp.zeros(C0.input_shape, C0.input_dtype)

        if self.admm.f is not None:
            ATWy = self.admm.f.A.adj(self.admm.f.W.diagonal * self.admm.f.y)  # type: ignore
            rhs += 2.0 * self.admm.f.scale * ATWy  # type: ignore

        for rhoi, Ci, zi, ui in zip(
            self.admm.rho_list, self.admm.C_list, self.admm.z_list, self.admm.u_list
        ):
            rhs += rhoi * Ci.adj(zi - ui)
        return rhs

    def solve(self, x0):
        """Solve the ADMM step.

        Args:
            x0: Initial value.

        Returns:
            Computed solution.
        """
        x0 = ensure_on_device(x0)
        rhs = self.compute_rhs()
        x, self.info = self.cg(self.lhs_op, rhs, x0, **self.cg_kwargs)  # type: ignore
        return x


######## Proximal Operators  ########
class ZeroOneIndicator(scico.functional.Functional):
    r"""Indicator function for [0,1] interval.

    Returns 0 if all elements of input array-like are between [0,1] and
    `inf` otherwise

    .. math::
        I(\mb{x}) = \begin{cases}
        0  & \text{ if } x_i \geq 0 and x_i \leq 1 \; \forall i \\
        \infty  & \text{ otherwise} \;.
        \end{cases}
    """

    has_eval = True
    has_prox = True

    def __call__(self, x):
        if snp.util.is_complex_dtype(x.dtype):
            raise ValueError("Not defined for complex input.")

        # Equivalent to snp.inf if snp.any(x < 0) else 0.0
        return snp.inf if snp.any(x<0) or snp.any(x>1) else 0.0


    def prox(
        self, v, lam: float = 1.0, **kwargs
    ):
        r"""The scaled proximal operator of the zero-one indicator.

        Evaluate the scaled proximal operator of the indicator over
        the [0,1] interval, :math:`I`,

        .. math::
            [\mathrm{prox}_{\lambda I}(\mb{v})]_i =
            \begin{cases}
            v_i\, & \text{ if } v_i \geq 0 and v_i \leq 1 \\
            1\, & \text{ if } v_i > 1 \\
            0\, & \text{ otherwise} \;.
            \end{cases}

        Args:
            v: Input array :math:`\mb{v}`.
            lam: Proximal parameter :math:`\lambda` (has no effect).
            kwargs: Additional arguments that may be used by derived
                classes.
        """
        return snp.minimum(snp.maximum(v, 0), 1)
