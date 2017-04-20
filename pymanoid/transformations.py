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
Conversion between rotations and homogeneous coordinates.

Rotations can be represented by:

- Roll-pitch-yaw angles, that is to say Euler angles corresponding to the
  sequence (1, 2, 3).
- Quaternions `[w x y z]`, with the scalar term `w` coming first following the
  OpenRAVE convention.
- Rotation matrices of shape :math:`3 \\times 3`.

Rigid-body transformations can be represented by:

- Poses in OpenRAVE format, which are simply 7D vectors consisting of the
  quaternion of the orientation followed by its position.
- Homogeneous transformation matrices `T` of shape :math:`4 \\times 4`.

Functions are provided to convert between all these representations. Most of
them are adapted from the comprehensive guide by Diebel [Diebel06]_.
"""

from math import asin, atan2, cos, sin
from numpy import array, dot, hstack, zeros
from openravepy import quatFromRotationMatrix as __quatFromRotationMatrix
from openravepy import rotationMatrixFromQuat as __rotationMatrixFromQuat


def crossmat(x):
    """
    Cross-product matrix of a 3D vector.

    Parameters
    ----------
    x : (3,) array
        Vector on the left-hand side of the cross product.

    Returns
    -------
    C : (3, 3) array
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
    quat : (4,) array
        Quaternion in `[w x y z]` format.

    Returns
    -------
    rpy : (3,) array
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


def quat_from_rpy(rpy):
    """
    Quaternion frmo roll-pitch-yaw angles.

    Parameters
    ----------
    rpy : (3,) array
        Vector of roll-pitch-yaw angles in [rad].

    Returns
    -------
    quat : (4,) array
        Quaternion in `[w x y z]` format.

    Notes
    -----
    Roll-pitch-yaw are Euler angles corresponding to the sequence (1, 2, 3).
    """
    roll, pitch, yaw = rpy
    cr, cp, cy = cos(roll / 2), cos(pitch / 2), cos(yaw / 2)
    sr, sp, sy = sin(roll / 2), sin(pitch / 2), sin(yaw / 2)
    return array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


def quat_from_rotation_matrix(R):
    """
    Quaternion from rotation matrix.

    Parameters
    ----------
    R : (3, 3) array
        Rotation matrix.

    Returns
    -------
    quat : (4,) array
        Quaternion in `[w x y z]` format.
    """
    return __quatFromRotationMatrix(R)


def rotation_matrix_from_quat(quat):
    """
    Rotation matrix from quaternion.

    Parameters
    ----------
    quat : (4,) array
        Quaternion in `[w x y z]` format.

    Returns
    -------
    R : (3, 3) array
        Rotation matrix.
    """
    return __rotationMatrixFromQuat(quat)


def rotation_matrix_from_rpy(rpy):
    """
    Rotation matrix from roll-pitch-yaw angles.

    Parameters
    ----------
    rpy : (3,) array
        Vector of roll-pitch-yaw angles in [rad].

    Returns
    -------
    R : (3, 3) array
        Rotation matrix.
    """
    return rotation_matrix_from_quat(quat_from_rpy(rpy))


def rpy_from_rotation_matrix(R):
    """
    Roll-pitch-yaw angles of rotation matrix.

    Parameters
    ----------
    R : array
        Rotation matrix.

    Returns
    -------
    rpy : (3,) array
        Array of roll-pitch-yaw angles, in [rad].

    Notes
    -----
    Roll-pitch-yaw are Euler angles corresponding to the sequence (1, 2, 3).
    """
    return rpy_from_quat(quat_from_rotation_matrix(R))


def transformation_from_pose(pose):
    """
    Transformation matrix from a pose vector.

    Parameters
    ----------
    pose : (7,) array
        Pose vector `[qw qx qy qz x y z]`.

    Returns
    -------
    T : (4, 4) array
        Homogeneous transformation matrix of the pose vector.
    """
    T = zeros((4, 4))
    T[:3, :3] = rotation_matrix_from_quat(pose[:4])
    T[:3, 3] = pose[4:]
    T[3, 3] = 1.
    return T


def pose_from_transformation(T):
    """
    Pose vector from a homogeneous transformation matrix.

    Parameters
    ----------
    T : (4, 4) array
        Homogeneous transformation matrix.

    Returns
    -------
    pose : (7,) array
        Pose vector `[qw qx qy qz x y z]` of the transformation matrix.
    """
    quat = quat_from_rotation_matrix(T[:3, :3])
    return hstack([quat, T[:3, 3]])


def transformation_inverse(T):
    """
    Inverse of a transformation matrix. Yields the same result but faster than
    :func:`numpy.linalg.inv` on such matrices.

    Parameters
    ----------
    T : (4, 4) array
        Homogeneous transformation matrix.

    Returns
    -------
    T_inv : (4, 4) array
        Inverse of `T`.
    """
    T_inv = zeros((4, 4))
    R_inv = T[:3, :3].T
    T_inv[:3, :3] = R_inv
    T_inv[:3, 3] = -dot(R_inv, T[:3, 3])
    T_inv[3, 3] = 1.
    return T_inv


__all__ = [
    'crossmat',
    'quat_from_rotation_matrix',
    'quat_from_rpy',
    'rpy_from_quat',
    'rpy_from_rotation_matrix',
    'rotation_matrix_from_quat',
    'rotation_matrix_from_rpy',
]
