#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
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
Most formulae in this module are adapted from the very useful document by James
Diebel:

"Representing Attitude: Euler Angles, Unit Quaternions, and Rotation Vectors"
http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.110.5134
"""


from math import atan2, asin, cos, sin, acos
from numpy import array, dot, sqrt
from openravepy import quatFromRotationMatrix


def crossmat(x):
    """Cross-product matrix of a 3D vector"""
    return array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]])


def rpy_from_quat(q):
    """Convention: upward pointing yaw axis."""
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
    """Convention: upward pointing yaw axis."""
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


quat_to_rot_tensor = array([[

    # [0, 0]: a^2 + b^2 - c^2 - d^2
    [[+1,  0,  0,  0],
     [.0, +1,  0,  0],
     [.0,  0, -1,  0],
     [.0,  0,  0, -1]],

    # [0, 1]: 2bc - 2ad
    [[.0,  0,  0, -2],
     [.0,  0, +2,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [0, 2]: 2bd + 2ac
    [[.0,  0, +2,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]]], [

    # [1, 0]: 2bc + 2ad
    [[.0,  0,  0, +2],
     [.0,  0, +2,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [1, 1]: a^2 - b^2 + c^2 - d^2
    [[+1,  0,  0,  0],
     [.0, -1,  0,  0],
     [.0,  0, +1,  0],
     [.0,  0,  0, -1]],

    # [1, 2]: 2cd - 2ab
    [[.0, -2,  0,  0],
     [.0,  0,  0,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0]]], [

    # [2, 0]: 2bd - 2ac
    [[.0,  0, -2,  0],
     [.0,  0,  0, +2],
     [.0,  0,  0,  0],
     [.0,  0,  0,  0]],

    # [2, 1]: 2cd + 2ab
    [[0, +2,  0,  0],
     [0,  0,  0,  0],
     [0,  0,  0, +2],
     [0,  0,  0,  0]],

    # [2, 2]: a^2 - b^2 - c^2 + d^2
    [[+1,  0,  0,  0],
     [.0, -1,  0,  0],
     [.0,  0, -1,  0],
     [.0,  0,  0, +1]]]])


def rotation_matrix_from_quat(quat):
    """
    For some reason, a bit faster than OpenRAVE's:

        In [13]: %timeit openravepy.rotationMatrixFromQuat([1, 0, 0, 0])
        100000 loops, best of 3: 6.67 µs per loop

        In [14]: %timeit rot_matrix_from_quat([1, 0, 0, 0])
        100000 loops, best of 3: 5.44 µs per loop

    """
    return dot(dot(quat_to_rot_tensor, quat), quat)


def quat_from_rotation_matrix(R):
    return quatFromRotationMatrix(R)


def rotation_matrix_from_rpy(roll, pitch, yaw):
    return rotation_matrix_from_quat(quat_from_rpy(roll, pitch, yaw))


def axis_angle_from_quat(quat):
    angle = 2 * acos(quat[0])
    axis = quat[1:] / sqrt(1. - quat[0] ** 2)
    return (axis, angle)


def axis_angle_from_rotation_matrix(R):
    return axis_angle_from_quat(quat_from_rotation_matrix(R))
