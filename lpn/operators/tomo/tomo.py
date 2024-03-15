import numpy as np
import sys
import scico
import os

# import odl from external source
# get current file path
_file_path = os.path.realpath(__file__)
sys.path.append(os.path.join(os.path.dirname(_file_path), "external", "odl"))
import odl


def get_operators(space_range, img_size, num_angles, det_shape):
    ##############compute projection#################
    space = odl.uniform_discr(
        [-space_range, -space_range],
        [space_range, space_range],
        (img_size, img_size),
        dtype="float32",
        weighting=1.0,
    )
    geometry = odl.tomo.geometry.parallel.parallel_beam_geometry(
        space, num_angles=num_angles, det_shape=det_shape
    )

    fwd_op_odl = odl.tomo.RayTransform(
        space, geometry, impl="astra_cuda"
    )  # astra_cuda, skimage
    fbp_op_odl = odl.tomo.fbp_op(fwd_op_odl)
    adjoint_op_odl = fwd_op_odl.adjoint

    def fwd_op_numpy(x):
        # print('fwd')
        return fwd_op_odl(np.array(x)).asarray()

    def adjoint_op_numpy(x):
        # print('adjoint')
        return adjoint_op_odl(np.array(x)).asarray()

    def fbp_op_numpy(x):
        # print('fbp')
        return fbp_op_odl(np.array(x)).asarray()

    return fwd_op_numpy, adjoint_op_numpy, fbp_op_numpy


def get_tomo_A(space_range, img_size, num_angles, det_shape):
    fwd_op_numpy, adjoint_op_numpy, fbp_op_numpy = get_operators(
        space_range, img_size, num_angles, det_shape
    )

    A = scico.linop.LinearOperator(
        input_shape=(img_size, img_size),
        output_shape=(num_angles, det_shape),
        eval_fn=fwd_op_numpy,
        adj_fn=adjoint_op_numpy,
        jit=False,
    )

    return A
