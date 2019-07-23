#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2018 Stephane Caron <stephane.caron@lirmm.fr>
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
# You should have received a copy of the GNU Lesser General Public License
# along with pymanoid. If not, see <http://www.gnu.org/licenses/>.

"""
Rotations and rigid-body transformations can be represented in many ways. For
rotations, the three main formats used in pymanoid are:

- **Roll-pitch-yaw angles:** that is to say Euler angles corresponding to the
  sequence (1, 2, 3).
- **Quaternions:** 4D vectors `[w x y z]`, with the scalar term `w` coming
  first following the OpenRAVE convention.
- **Rotation matrices:** :math:`3 \\times 3` matrices :math:`R` whose inverse
  is equal to their transpose.

Rigid-body transformations can be represented by:

- **Poses:** 7D vectors consisting of the quaternion of the orientation
  followed by its position.
- **Transformation matrices:** :math:`4 \\times 4` matrices :math:`T`.

Functions are provided to convert between all these representations. Most of
them are adapted from the comprehensive guide [Diebel06]_.
"""

from math import asin, atan2, cos, sin
from numpy import array, cross, dot, eye, hstack, zeros
from openravepy import quatFromRotationMatrix as __quatFromRotationMatrix
from openravepy import rotationMatrixFromQuat as __rotationMatrixFromQuat
from openravepy import axisAngleFromQuat as axis_angle_from_quat

from .misc import norm


def apply_transform(T, p):
    """
    Apply a transformation matrix `T` to point coordinates `p`.

    Parameters
    ----------
    T : (4, 4) array
        Homogeneous transformation matrix.
    p : (7,) or (3,) array
        Pose or point coordinates.

    Returns
    -------
    Tp : (7,) or (3,) array
        Result after applying the transformation.

    Notes
    -----
    For a single point, it is faster to apply the matrix multiplication
    directly rather than calling the OpenRAVE function:

    .. code:: python

        In [33]: %timeit dot(T, hstack([p, 1]))[:3]
        100000 loops, best of 3: 3.82 µs per loop

        In [34]: %timeit list(transformPoints(T, [p]))
        100000 loops, best of 3: 6.4 µs per loop
    """
    if len(p) == 3:
        return dot(T, hstack([p, 1]))[:3]
    R = dot(T[:3, :3], rotation_matrix_from_quat(p[:4]))
    pos = dot(T, hstack([p[4:], 1]))[:3]
    return hstack([quat_from_rotation_matrix(R), pos])


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


def integrate_angular_acceleration(R, omega, omegad, dt):
    """
    Integrate constant angular acceleration :math:`\\dot{\\omega}` for a
    duration `dt` starting from the orientation :math:`R` and angular velocity
    :math:`\\omega`.

    Parameters
    ----------
    R : (3, 3) array
        Rotation matrix.
    omega : (3,) array
        Initial angular velocity.
    omegad : (3,) array
        Constant angular acceleration.
    dt : scalar
        Duration.

    Returns
    -------
    Ri : (3, 3) array
        Rotation matrix after integration.

    Notes
    -----
    This function applies `Rodrigues' rotation formula
    <https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula>`_. See also
    the following nice post on `integrating rotations
    <https://cwzx.wordpress.com/2013/12/16/numerical-integration-for-rotational-dynamics/>`_.
    """
    Omega = magnus_expansion(omega, omegad, dt)
    theta = norm(Omega)
    if theta < 1e-5:
        return eye(3)
    K = crossmat(Omega / theta)
    exp_Omega = eye(3) + sin(theta) * K + (1. - cos(theta)) * dot(K, K)
    return dot(exp_Omega, R)


def integrate_body_acceleration(T, v, a, dt):
    """
    Integrate constant body acceleration :math:`a` for a duration `dt` starting
    from the affine transform :math:`T` and body velocity :math:`v`.

    Parameters
    ----------
    T : (4, 4) array
        Affine transform of the rigid body.
    v : (3,) array
        Initial body velocity of the rigid body.
    a : (3,) array
        Constant body acceleration.
    dt : scalar
        Duration.

    Returns
    -------
    T_f : (3, 3) array
        Affine transform after integration.

    Notes
    -----
    With full frame notations, the affine transform :math:`T = {}^W T_B` maps
    vectors from the rigid body frame :math:`B` to the inertial frame
    :math:`W`. Its body velocity :math:`v = \\begin{bmatrix} v_B & \\omega
    \\end{bmatrix}` is a twist in the inertial frame, with coordinates taken at
    the origin of :math:`B`. The same goes for its body acceleration :math:`a =
    \\begin{bmatrix} a_B & \\dot{\\omega} \\end{bmatrix}`.
    """
    R = T[0:3, 0:3]
    p = T[0:3, 3]
    omega = v[3:6]
    omegad = a[3:6]
    R_f = integrate_angular_acceleration(R, omega, omegad, dt)
    v_f = v + a * dt
    p_f = p + v_f[0:3] * dt  # semi-implicit Euler integration
    T_f = eye(4)
    T_f[0:3, 0:3] = R_f
    T_f[0:3, 3] = p_f
    return T_f


def magnus_expansion(omega, omegad, dt):
    """
    Compute the integral :math:`\\Omega` obtained by integrating
    :math:`\\dot{\\omega}` for a duration `dt` starting from :math:`\\omega`.

    Parameters
    ----------
    omega : (3,) array
        Initial angular velocity.
    omegad : (3,) array
        Constant angular acceleration.
    dt : scalar
        Duration.

    Returns
    -------
    Omega : (3,) array
        Integral of ``omegad`` for ``dt`` starting from ``omega``.

    Notes
    -----
    This function only computes the first three terms of the Magnus expansion.
    See <https://github.com/jrl-umi3218/RBDyn/pull/66/files> if you need a more
    advanced version.
    """
    dt_sq = dt * dt
    omega1 = omega + omegad * dt
    Omega1 = (omega + omega1) * dt / 2.
    Omega2 = cross(omega1, omega) * dt_sq / 12.
    Omega3 = cross(omegad, Omega2) * dt_sq / 20.
    return Omega1 + Omega2 + Omega3


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
    cr, cp, cy = cos(roll / 2.), cos(pitch / 2.), cos(yaw / 2.)
    sr, sp, sy = sin(roll / 2.), sin(pitch / 2.), sin(yaw / 2.)
    return array([
        cr * cp * cy + sr * sp * sy,
        -cr * sp * sy + cp * cy * sr,
        cr * cy * sp + sr * cp * sy,
        cr * cp * sy - sr * cy * sp])


def pose_from_transform(T):
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


def transform_from_pose(pose):
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


def transform_from_R_p(R, p):
    """
    Transformation matrix from a translation and rotation matrix.

    Parameters
    ----------
    R : (3, 3) array
        Rotation matrix.
    p : (3,) array
        Point coordinates.

    Returns
    -------
    T : (4, 4) array
        Homogeneous transformation matrix.
    """
    T = eye(4)
    T[:3, :3] = R
    T[:3, 3] = p
    return T


def transform_inverse(T):
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
    'apply_transform',
    'axis_angle_from_quat',
    'crossmat',
    'integrate_angular_acceleration',
    'integrate_body_acceleration',
    'magnus_expansion',
    'pose_from_transform',
    'quat_from_rotation_matrix',
    'quat_from_rpy',
    'rotation_matrix_from_quat',
    'rotation_matrix_from_rpy',
    'rpy_from_quat',
    'rpy_from_rotation_matrix',
    'transform_from_pose',
    'transform_inverse',
]
