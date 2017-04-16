#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import arange, dot, hstack
from openravepy import InterpolateQuatSlerp as quat_slerp

from draw import draw_trajectory
from misc import NDPolynomial
from rotations import rotation_matrix_from_quat


def interpolate_cubic_hermite(p0, v0, p1, v1):
    """
    Compute the third-order polynomial of the Hermite curve connecting
    :math:`(p_0, v_0)` to :math:`(p_1, v_1)`.

    Parameters
    ----------
    p0 : (3,) array
        Start point.
    v0 : (3,) array
        Start velocity.
    p1 : (3,) array
        End point.
    v1 : (3,) array
        End velocity.

    Returns
    -------
    P : NDPolynomial
        Polynomial function of the Hermite curve.
    """
    C3 = 2 * p0 - 2 * p1 + v0 + v1
    C2 = -3 * p0 + 3 * p1 - 2 * v0 - v1
    C1 = v0
    C0 = p0
    return NDPolynomial([C0, C1, C2, C3])


def interpolate_pose_linear(pose0, pose1, x):
    """
    Standalone function for linear pose interpolation.

    Parameters
    ----------
    pose0 : (7,) array
        First pose.
    pose1 : (7,) array
        Second pose.
    x : scalar
        Number between 0 and 1.

    Returns
    -------
    pose : (7,) array
        Linear interpolation between the first two arguments.
    """
    pos = pose0[4:] + x * (pose1[4:] - pose0[4:])
    quat = quat_slerp(pose0[:4], pose1[:4], x)
    return hstack([quat, pos])


def interpolate_pose_quadratic(pose0, pose1, x):
    """
    Pose interpolation that is linear in orientation (SLERP) and
    quadratic (Bezier) in position.

    Parameters
    ----------
    pose0 : (7,) array
        First pose.
    pose1 : (7,) array
        Second pose.
    x : scalar
        Number between 0 and 1.

    Returns
    -------
    pose : (7,) array
        Linear interpolation between the first two arguments.

    Note
    ----
    Initial and final linear velocities on the interpolated trajectory are zero.
    """
    pos = x ** 2 * (3 - 2 * x) * (pose1[4:] - pose0[4:]) + pose0[4:]
    quat = quat_slerp(pose0[:4], pose1[:4], x)
    return hstack([quat, pos])


def interpolate_uab_hermite(p0, u0, p1, u1):
    """
    Interpolate a Hermite path between :math:`p_0` and :math:`p_1` with tangents
    parallel to :math:`u_0` and :math:`u_1`, respectively. The output path
    `B(s)` minimizes a relaxation of the uniform acceleration bound:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & M \\\\
        \\mathrm{subject\\ to} & & \\forall s \\in [0, 1],\\
            \\|\\ddot{B}(s)\\|^2 \\leq M
        \\end{eqnarray}

    Parameters
    ----------
    p0 : (3,) array
        Start point.
    u0 : (3,) array
        Start tangent.
    p1 : (3,) array
        End point.
    u1 : (3,) array
        End tangent.

    Returns
    -------
    P : numpy.polynomial.Polynomial
        Polynomial function of the Hermite curve.

    Note
    ----
    We also impose that the output tangents share the sign of :math:`t_0` and
    :math:`t_1`, respectively.
    """
    Delta = p1 - p0
    _Delta_u0 = dot(Delta, u0)
    _Delta_u1 = dot(Delta, u1)
    _u0_u0 = dot(u0, u0)
    _u0_u1 = dot(u0, u1)
    _u1_u1 = dot(u1, u1)
    b0 = 6 * (3 * _Delta_u0 * _u1_u1 - 2 * _Delta_u1 * _u0_u1) / (
        9 * _u0_u0 * _u1_u1 - 4 * _u0_u1 * _u0_u1)
    if b0 < 0:
        b0 *= -1
    b1 = 6 * (-2 * _Delta_u0 * _u0_u1 + 3 * _Delta_u1 * _u0_u0) / (
        9 * _u0_u0 * _u1_u1 - 4 * _u0_u1 * _u0_u1)
    if b1 < 0:
        b1 *= -1
    return interpolate_cubic_hermite(p0, b0 * u0, p1, b1 * u1)


class PoseInterpolator(object):

    def __init__(self, start_pose, end_pose):
        self.end_quat = end_pose[:4]
        self.start_quat = start_pose[:4]

    def eval_quat(self, x):
        return quat_slerp(self.start_quat, self.end_quat, x)

    def eval_pos(self, x):
        raise NotImplementedError()

    def __call__(self, x):
        return hstack([self.eval_quat(x), self.eval_pos(x)])


class LinearPoseInterpolator(PoseInterpolator):

    def __init__(self, start_pose, end_pose):
        assert len(start_pose) == len(end_pose) == 7
        super(LinearPoseInterpolator, self).__init__(start_pose, end_pose)
        self.delta_pos = end_pose[4:] - start_pose[4:]
        self.start_pos = start_pose[4:]

    def eval_pos(self, x):
        return self.start_pos + x * self.delta_pos


class LinearPosInterpolator(LinearPoseInterpolator):

    def __init__(self, start_pos, end_pos):
        assert len(start_pos) == len(end_pos) == 3
        self.delta_pos = end_pos - start_pos
        self.start_pos = start_pos

    def __call__(self, x):
        return self.eval_pos(x)


class QuadraticPoseInterpolator(PoseInterpolator):

    def __init__(self, start_pose, end_pose):
        assert len(start_pose) == len(end_pose) == 7
        super(QuadraticPoseInterpolator, self).__init__(start_pose, end_pose)
        self.delta_pos = end_pose[4:] - start_pose[4:]
        self.start_pos = start_pose[4:]

    def eval_pos(self, x):
        return self.start_pos + x ** 2 * (3 - 2 * x) * self.delta_pos


class QuadraticPosInterpolator(QuadraticPoseInterpolator):

    def __init__(self, start_pos, end_pos):
        assert len(start_pos) == len(end_pos) == 3
        self.delta_pos = end_pos - start_pos
        self.start_pos = start_pos

    def __call__(self, x):
        return self.eval_pos(x)


class SwingFootInterpolator(PoseInterpolator):

    def __init__(self, start_pose, end_pose):
        super(SwingFootInterpolator, self).__init__(start_pose, end_pose)
        p0 = start_pose[4:]
        p1 = end_pose[4:]
        R0 = rotation_matrix_from_quat(self.start_quat)
        R1 = rotation_matrix_from_quat(self.end_quat)
        u0 = dot(R0, [0.3, 0., 0.7])
        u1 = dot(R1, [0.5, 0., -0.5])
        self.poly = interpolate_uab_hermite(p0, u0, p1, u1)

    def eval_pos(self, x):
        return self.poly(x)

    def draw(self, nb_segments=15, color='b'):
        """
        Draw the interpolated foot path.

        Parameters
        ----------
        nb_segments : scalar
            Number of segments approximating the polynomial curve.
        color : char or triplet, optional
            Color letter or RGB values, default is 'b' for green.

        Returns
        -------
        handle : openravepy.GraphHandle
            OpenRAVE graphical handle. Must be stored in some variable,
            otherwise the drawn object will vanish instantly.
        """
        dx = 1. / nb_segments
        points = [self.poly(x) for x in arange(0., 1. + dx, dx)]
        return draw_trajectory(points, color=color)
