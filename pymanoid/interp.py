#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import hstack, linspace
from openravepy import InterpolateQuatSlerp as quat_slerp

from gui import draw_trajectory
from misc import NDPolynomial


def interpolate_cubic_bezier(p0, p1, p2, p3):
    """
    Compute the cubic Bezier curve corresponding to four control points.

    Parameters
    ----------
    p0 : (3,) array
        First control point.
    p1 : (3,) array
        Second control point.
    p2 : (3,) array
        Third control point.
    p3 : (3,) array
        Fourth control point.

    Returns
    -------
    P : NDPolynomial
        Polynomial function of the Bezier curve.
    """
    c3 = p3 - 3 * (p2 - p1) - p0
    c2 = 3 * (p2 - 2 * p1 + p0)
    c1 = 3 * (p1 - p0)
    c0 = p0
    return NDPolynomial([c0, c1, c2, c3])


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
    c3 = 2 * p0 - 2 * p1 + v0 + v1
    c2 = -3 * p0 + 3 * p1 - 2 * v0 - v1
    c1 = v0
    c0 = p0
    return NDPolynomial([c0, c1, c2, c3])


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


class PoseInterpolator(object):

    """
    Interpolate a body trajectory between two poses. Orientations are
    computed by `spherical linear interpolation
    <https://en.wikipedia.org/wiki/Slerp>`_.

    Parameters
    ----------
    start_pose : (7,) array
        Initial pose to interpolate from.
    end_pose : (7,) array
        End pose to interpolate to.
    body : Body, optional
        Body affected by the update function.
    """

    def __init__(self, start_pose, end_pose, duration, body=None):
        self.body = body
        self.duration = duration
        self.end_quat = end_pose[:4]
        self.start_quat = start_pose[:4]

    def eval_quat(self, s):
        return quat_slerp(self.start_quat, self.end_quat, s)

    def eval_pos(self, s):
        """
        Evaluate position at index s between 0 and 1.

        Parameters
        ----------
        s : scalar
            Normalized trajectory index between 0 and 1.
        """
        raise NotImplementedError()

    def __call__(self, t):
        s = 0. if t < 0. else 1. if t > self.duration else t / self.duration
        return hstack([self.eval_quat(s), self.eval_pos(s)])

    def update(self, t):
        pose = self.__call__(t)
        self.body.set_pose(pose)

    def draw(self, color='b'):
        """
        Draw positions of the interpolated trajectory.

        Parameters
        ----------
        color : char or triplet, optional
            Color letter or RGB values, default is 'b' for green.

        Returns
        -------
        handle : openravepy.GraphHandle
            OpenRAVE graphical handle. Must be stored in some variable,
            otherwise the drawn object will vanish instantly.
        """
        points = [self.eval_pos(s) for s in linspace(0, 1, 10)]
        return draw_trajectory(points, color=color)


class LinearPoseInterpolator(PoseInterpolator):

    """
    Interpolate a body trajectory between two poses. Intermediate positions are
    interpolated linearly, while orientations are computed by `spherical linear
    interpolation <https://en.wikipedia.org/wiki/Slerp>`_.

    Parameters
    ----------
    start_pose : (7,) array
        Initial pose to interpolate from.
    end_pose : (7,) array
        End pose to interpolate to.
    body : Body, optional
        Body affected by the update function.
    """

    def __init__(self, start_pose, end_pose, duration, body=None):
        assert len(start_pose) == len(end_pose) == 7
        super(LinearPoseInterpolator, self).__init__(
            start_pose, end_pose, duration, body)
        self.delta_pos = end_pose[4:] - start_pose[4:]
        self.start_pos = start_pose[4:]

    def eval_pos(self, s):
        """
        Evaluate position at index s between 0 and 1.

        Parameters
        ----------
        s : scalar
            Normalized trajectory index between 0 and 1.
        """
        return self.start_pos + s * self.delta_pos


class CubicPoseInterpolator(PoseInterpolator):

    """
    Interpolate a body trajectory between two poses. Intermediate positions are
    interpolated cubically with zero boundary velocities, while orientations
    are computed by `spherical linear interpolation
    <https://en.wikipedia.org/wiki/Slerp>`_.

    Parameters
    ----------
    start_pose : (7,) array
        Initial pose to interpolate from.
    end_pose : (7,) array
        End pose to interpolate to.
    body : Body, optional
        Body affected by the update function.
    """

    def __init__(self, start_pose, end_pose, duration, body=None):
        assert len(start_pose) == len(end_pose) == 7
        super(CubicPoseInterpolator, self).__init__(
            start_pose, end_pose, duration, body)
        self.delta_pos = end_pose[4:] - start_pose[4:]
        self.start_pos = start_pose[4:]

    def eval_pos(self, s):
        """
        Evaluate position at index s between 0 and 1.

        Parameters
        ----------
        s : scalar
            Normalized trajectory index between 0 and 1.
        """
        return self.start_pos + s ** 2 * (3 - 2 * s) * self.delta_pos


class QuinticPoseInterpolator(PoseInterpolator):

    """
    Interpolate a body trajectory between two poses. Intermediate positions are
    interpolated quintically with zero-velocity and zero-acceleration boundary
    conditions, while orientations are computed by `spherical linear
    interpolation <https://en.wikipedia.org/wiki/Slerp>`_.

    Parameters
    ----------
    start_pose : (7,) array
        Initial pose to interpolate from.
    end_pose : (7,) array
        End pose to interpolate to.
    body : Body, optional
        Body affected by the update function.
    """

    def __init__(self, start_pose, end_pose, duration, body=None):
        assert len(start_pose) == len(end_pose) == 7
        super(QuinticPoseInterpolator, self).__init__(
            start_pose, end_pose, duration, body)
        self.delta_pos = end_pose[4:] - start_pose[4:]
        self.start_pos = start_pose[4:]

    def eval_pos(self, s):
        """
        Evaluate position at index s between 0 and 1.

        Parameters
        ----------
        s : scalar
            Normalized trajectory index between 0 and 1.
        """
        return self.start_pos + \
            s ** 3 * (10 - 15 * s + 6 * s ** 2) * self.delta_pos


class LinearPosInterpolator(LinearPoseInterpolator):

    """
    Linear interpolation between two positions.

    Parameters
    ----------
    start_pos : (3,) array
        Initial position to interpolate from.
    end_pos : (3,) array
        End position to interpolate to.
    body : Body, optional
        Body affected by the update function.
    """

    def __init__(self, start_pos, end_pos, duration, body=None):
        # parent constructor is not called
        assert len(start_pos) == len(end_pos) == 3
        self.body = body
        self.delta_pos = end_pos - start_pos
        self.duration = duration
        self.start_pos = start_pos

    def __call__(self, t):
        s = t / self.duration
        return self.eval_pos(s)


class CubicPosInterpolator(CubicPoseInterpolator):

    """
    Cubic interpolation between two positions with zero-velocity boundary
    conditions.

    Parameters
    ----------
    start_pos : (3,) array
        Initial position to interpolate from.
    end_pos : (3,) array
        End position to interpolate to.
    body : Body, optional
        Body affected by the update function.
    """

    def __init__(self, start_pos, end_pos, duration, body=None):
        # parent constructor is not called
        assert len(start_pos) == len(end_pos) == 3
        self.body = body
        self.delta_pos = end_pos - start_pos
        self.duration = duration
        self.start_pos = start_pos

    def __call__(self, t):
        s = t / self.duration
        return self.eval_pos(s)


class QuinticPosInterpolator(QuinticPoseInterpolator):

    """
    Quintic interpolation between two positions with zero-velocity and
    zero-acceleration boundary conditions.

    Parameters
    ----------
    start_pos : (3,) array
        Initial position to interpolate from.
    end_pos : (3,) array
        End position to interpolate to.
    body : Body, optional
        Body affected by the update function.
    """

    def __init__(self, start_pos, end_pos, duration, body=None):
        # parent constructor is not called
        assert len(start_pos) == len(end_pos) == 3
        self.body = body
        self.delta_pos = end_pos - start_pos
        self.duration = duration
        self.start_pos = start_pos

    def __call__(self, t):
        s = t / self.duration
        return self.eval_pos(s)
