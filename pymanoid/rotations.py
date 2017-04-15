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

"""
This module mostly imports top-level rotation functions from OpenRAVE in order
to make them visible in pymanoid. Conversions to and from roll-pitch-yaw (in
humanoid-robotics convention: with an upward yaw axis) were adapted from
[Die+06]_.
"""

from math import asin, atan2, cos, sin
from numpy import array
from openravepy import \
    axisAngleFromQuat as axis_angle_from_quat, \
    axisAngleFromRotationMatrix as axis_angle_from_rotation_matrix, \
    quatFromRotationMatrix as quat_from_rotation_matrix, \
    quatInverse as quat_inverse, \
    quatMultiply as quat_multiply, \
    rotationMatrixFromQuat as rotation_matrix_from_quat


def crossmat(x):
    """
    Cross-product matrix of a 3D vector.

    Parameters
    ----------
    x : array, shape=(3,)
        Vector on the left-hand side of the cross product.

    Returns
    -------
    C : array, shape=(3, 3)
        Cross-product matrix of `x`.
    """
    return array([
        [0., -x[2], x[1]],
        [x[2], 0., -x[0]],
        [-x[1], x[0], 0.]])


def rpy_from_quat(quat):
    """
    Roll-pitch-yaw angles of a quaternion.

    Parameters
    ----------
    quat : array, shape=(4,)
        Quaternion in `[w x y z]` format.

    Returns
    -------
    rpy : array, shape=(3,)
        Array of roll-pitch-yaw angles, in [rad].

    Notes
    -----
    Roll-pitch-yaw are Euler angles corresponding to the sequence (1, 2, 3).
    """
    roll = atan2(
        2 * quat[2] * quat[3] + 2 * quat[0] * quat[1],
        quat[3] ** 2 - quat[2] ** 2 - quat[1] ** 2 + quat[0] ** 2)
    pitch = -asin(
        2 * quat[1] * quat[3] - 2 * quat[0] * quat[2])
    yaw = atan2(
        2 * quat[1] * quat[2] + 2 * quat[0] * quat[3],
        quat[1] ** 2 + quat[0] ** 2 - quat[3] ** 2 - quat[2] ** 2)
    return array([roll, pitch, yaw])


def quat_from_rpy(roll, pitch, yaw):
    """
    Quaternion frmo roll-pitch-yaw angles.

    Parameters
    ----------
    roll : scalar
        Roll angle in [rad].
    pitch : scalar
        Pitch angle in [rad].
    yaw : scalar
        Yaw angle in [rad].

    Returns
    -------
    quat : array, shape=(4,)
        Quaternion in `[w x y z]` format.

    Notes
    -----
    Roll-pitch-yaw are Euler angles corresponding to the sequence (1, 2, 3).
    """
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


def rotation_matrix_from_rpy(roll, pitch, yaw):
    """
    Rotation matrix from roll-pitch-yaw angles.

    Parameters
    ----------
    roll : scalar
        Roll angle in [rad].
    pitch : scalar
        Pitch angle in [rad].
    yaw : scalar
        Yaw angle in [rad].

    Returns
    -------
    R : array, shape=(3, 3)
        Rotation matrix.
    """
    return rotation_matrix_from_quat(quat_from_rpy(roll, pitch, yaw))


def rpy_from_rotation_matrix(R):
    """
    Roll-pitch-yaw angles of rotation matrix.

    Parameters
    ----------
    R : array
        Rotation matrix.

    Returns
    -------
    rpy : array, shape=(3,)
        Array of roll-pitch-yaw angles, in [rad].

    Notes
    -----
    Roll-pitch-yaw are Euler angles corresponding to the sequence (1, 2, 3).
    """
    return rpy_from_quat(quat_from_rotation_matrix(R))


__all__ = [
    'axis_angle_from_quat',
    'axis_angle_from_rotation_matrix',
    'crossmat',
    'quat_from_rotation_matrix',
    'quat_from_rpy',
    'quat_inverse',
    'quat_multiply',
    'rpy_from_quat',
    'rpy_from_rotation_matrix',
    'rotation_matrix_from_quat',
    'rotation_matrix_from_rpy',
]
