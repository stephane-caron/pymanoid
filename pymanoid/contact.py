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

from numpy import array, cross, dot, eye, hstack, sqrt, vstack
from scipy.linalg import block_diag

from body import Box
from polyhedra import Cone
from rotations import crossmat


class Contact(Box):

    """
    Rectangular contact surface.

    Parameters
    ----------
    shape : double couple
        Surface dimensions (half-length, half-width) in [m].
    pos : ndarray
        Contact position in world frame.
    rpy : ndarray
        Contact orientation in world frame.
    pose : ndarray
        Initial pose. Supersedes ``pos`` and ``rpy`` if they are provided at the
        same time.
    static_friction : double
        Static friction coefficient.
    kinetic_friction : double
        Kinetic friction coefficient.
    visible : bool, optional
        Initial visibility. Defaults to ``True``.
    name : string, optional
        Name in OpenRAVE scope.
    """

    THICKNESS = 0.01
    """Thickness in [m] of the contact patch."""

    def __init__(self, shape, pos=None, rpy=None, pose=None,
                 static_friction=None, kinetic_friction=None, visible=True,
                 name=None, color='r'):
        X, Y = shape
        super(Contact, self).__init__(
            X, Y, Z=self.THICKNESS, pos=pos, rpy=rpy, pose=pose,
            visible=visible, dZ=-self.THICKNESS, name=name)
        if kinetic_friction is None and static_friction is not None:
            kinetic_friction = static_friction
        self.kinetic_friction = kinetic_friction
        self.shape = shape
        self.static_friction = static_friction

    """
    Geometry
    ========
    """

    def grasp_matrix(self, p):
        """Compute the grasp matrix for a given destination point.

        The grasp matrix :math:`G_P` converts the local contact wrench :math:`w`
        to the contact wrench :math:`w_P` at another point :math:`P`:

        .. math::

            w_P = G_P w

        All wrenches are expressed with respect to the world frame.

        Parameters
        ----------
        p : ndarray
            Point, in world frame coordinates, where the wrench is taken.

        Returns
        -------
        G : ndarray
            Grasp matrix :math:`G_P`.
        """
        x, y, z = self.p - p
        return array([
            # fx fy  fz taux tauy tauz
            [1,   0,  0,   0,   0,   0],
            [0,   1,  0,   0,   0,   0],
            [0,   0,  1,   0,   0,   0],
            [0,  -z,  y,   1,   0,   0],
            [z,   0, -x,   0,   1,   0],
            [-y,  x,  0,   0,   0,   1]])

    @property
    def vertices(self):
        """Vertices of the contact area."""
        c1 = dot(self.T, array([+self.X, +self.Y, -self.Z, 1.]))[:3]
        c2 = dot(self.T, array([+self.X, -self.Y, -self.Z, 1.]))[:3]
        c3 = dot(self.T, array([-self.X, -self.Y, -self.Z, 1.]))[:3]
        c4 = dot(self.T, array([-self.X, +self.Y, -self.Z, 1.]))[:3]
        return [c1, c2, c3, c4]

    """
    Force Friction Cone
    ===================

    All linearized friction cones in pymanoid use the inner (conservative)
    approximation. See <https://scaron.info/teaching/friction-model.html>
    """

    @property
    def force_cone(self):
        """
        Contact force friction cone.
        """
        return Cone(face=self.force_face, rays=self.force_rays)

    @property
    def force_face(self):
        """
        Face (H-rep) of the force friction cone in world frame.
        """
        mu = self.static_friction / sqrt(2)  # inner approximation
        local_cone = array([
            [-1, 0, -mu],
            [+1, 0, -mu],
            [0, -1, -mu],
            [0, +1, -mu]])
        return dot(local_cone, self.R.T)

    @property
    def force_rays(self):
        """
        Rays (V-rep) of the force friction cone in world frame.
        """
        mu = self.static_friction / sqrt(2)  # inner approximation
        f1 = dot(self.R, [+mu, +mu, +1])
        f2 = dot(self.R, [+mu, -mu, +1])
        f3 = dot(self.R, [-mu, +mu, +1])
        f4 = dot(self.R, [-mu, -mu, +1])
        return [f1, f2, f3, f4]

    @property
    def force_span(self):
        """
        Span matrix of the force friction cone in world frame.
        """
        return array(self.force_rays).T

    """
    Wrench Friction Cone
    ====================
    """

    @property
    def wrench_cone(self):
        """
        Contact wrench friction cone (CWC).
        """
        return Cone(face=self.wrench_face, rays=self.wrench_rays)

    @property
    def wrench_face(self):
        """Matrix :math:`F` of static-friction inequalities in world frame.

        This matrix describes the linearized Coulomb friction model by:

        .. math::

            F w \leq 0

        where `w` is the contact wrench at the contact point (``self.p``) in the
        world frame. See [CPN15]_ for the derivation of the formula for `F`.

        References
        ----------
        .. [CPN15] Caron, Pham, Nakamura, "Stability of surface contacts for
            humanoid robots: Closed-form formulae of the contact wrench cone for
            rectangular support areas." 2015 IEEE International Conference on
            Robotics and Automation (ICRA).
            `[doi] <http://doai.io/10.1109/ICRA.2015.7139910>`__
            `[pdf] <https://scaron.info/papers/conf/caron-icra-2015.pdf>`__
        """
        X, Y = self.X, self.Y
        mu = self.static_friction / sqrt(2)  # inner approximation
        local_cone = array([
            # fx fy             fz taux tauy tauz
            [-1,  0,           -mu,   0,   0,   0],
            [+1,  0,           -mu,   0,   0,   0],
            [0,  -1,           -mu,   0,   0,   0],
            [0,  +1,           -mu,   0,   0,   0],
            [0,   0,            -Y,  -1,   0,   0],
            [0,   0,            -Y,  +1,   0,   0],
            [0,   0,            -X,   0,  -1,   0],
            [0,   0,            -X,   0,  +1,   0],
            [-Y, -X, -(X + Y) * mu, +mu, +mu,  -1],
            [-Y, +X, -(X + Y) * mu, +mu, -mu,  -1],
            [+Y, -X, -(X + Y) * mu, -mu, +mu,  -1],
            [+Y, +X, -(X + Y) * mu, -mu, -mu,  -1],
            [+Y, +X, -(X + Y) * mu, +mu, +mu,  +1],
            [+Y, -X, -(X + Y) * mu, +mu, -mu,  +1],
            [-Y, +X, -(X + Y) * mu, -mu, +mu,  +1],
            [-Y, -X, -(X + Y) * mu, -mu, -mu,  +1]])
        return dot(local_cone, block_diag(self.R.T, self.R.T))

    @property
    def wrench_rays(self):
        """
        Rays (V-rep) of the contact wrench cone in world frame.
        """
        rays = []
        for v in self.vertices:
            x, y, z = v - self.p
            for f in self.force_rays:
                rays.append(hstack([f, cross(v - self.p, f)]))
        return rays

    @property
    def wrench_span(self):
        """
        Span matrix of the contact wrench cone in world frame.

        This matrix is such that all valid contact wrenches can be written as:

        .. math::

            w_P = S \\lambda, \\quad \\lambda \\geq 0

        where `S` is the friction span and :math:`\\lambda` is a vector with
        positive coordinates.

        Returns
        -------
        S : array, shape=(6, 16)
            Span matrix of the contact wrench cone.

        Notes
        -----
        Note that the contact wrench coordinates :math:`w_P` ("output" of `S`)
        are taken at the contact point `P` (``self.p``) and in the world frame.
        Meanwhile, the number of columns of `S` results from our choice of 4
        contact points (one for each vertex of the rectangular area) with
        4-sided friction pyramids at each.
        """
        span_blocks = []
        for v in self.vertices:
            G = vstack([eye(3), crossmat(v - self.p)])
            span_blocks.append(dot(G, self.force_span))
        return hstack(span_blocks)
