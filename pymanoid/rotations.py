#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

"""
This module mostly imports top-level rotation functions from OpenRAVE in order
to make them visible in pymanoid. Conversions to and from roll-pitch-yaw (in
humanoid-robotics convention: with an upward yaw axis) were adapted from a
useful document by James Diebel:

    Representing Attitude: Euler Angles, Unit Quaternions, and Rotation
    Vectors <http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134>
"""

from math import asin, atan2, cos, sin
from numpy import array
from openravepy import \
    InterpolateQuatSlerp as quat_slerp, \
    axisAngleFromQuat as axis_angle_from_quat, \
    axisAngleFromRotationMatrix as axis_angle_from_rotation_matrix, \
    quatFromRotationMatrix as quat_from_rotation_matrix, \
    quatInverse as quat_inverse, \
    quatMultiply as quat_multiply, \
    rotationMatrixFromQuat as rotation_matrix_from_quat


def crossmat(x):
    """Cross-product matrix of a 3D vector"""
    return array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]])


def rpy_from_quat(q):
    """Roll-pitch-yaw is Euler sequence (1, 2, 3)."""
    roll = atan2(
        2 * q[2] * q[3] + 2 * q[0] * q[1],
        q[3] ** 2 - q[2] ** 2 - q[1] ** 2 + q[0] ** 2)
    pitch = -asin(
        2 * q[1] * q[3] - 2 * q[0] * q[2])
    yaw = atan2(
        2 * q[1] * q[2] + 2 * q[0] * q[3],
        q[1] ** 2 + q[0] ** 2 - q[3] ** 2 - q[2] ** 2)
    return array([roll, pitch, yaw])


def quat_from_rpy(roll, pitch, yaw):
    """Roll-pitch-yaw is Euler sequence (1, 2, 3)."""
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


def rotation_matrix_from_rpy(roll, pitch, yaw):
    return rotation_matrix_from_quat(quat_from_rpy(roll, pitch, yaw))


def rpy_from_rotation_matrix(R):
    return rpy_from_quat(quat_from_rotation_matrix(R))


__all__ = [
    'axis_angle_from_quat',
    'axis_angle_from_rotation_matrix',
    'crossmat',
    'quat_from_rotation_matrix',
    'quat_from_rpy',
    'quat_inverse',
    'quat_multiply',
    'quat_slerp',
    'rpy_from_quat',
    'rpy_from_rotation_matrix',
    'rotation_matrix_from_quat',
    'rotation_matrix_from_rpy',
]
