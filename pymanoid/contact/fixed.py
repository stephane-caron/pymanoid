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

from numpy import array, cross, dot, eye, hstack, sqrt, vstack
from scipy.linalg import block_diag

from generic import Contact
from pymanoid.rotations import crossmat


class FixedContact(Contact):

    """
    Force Friction Cone
    ===================

    All linearized friction cones in pymanoid use the inner (conservative)
    approximation. See <https://scaron.info/teaching/friction-model.html>
    """

    @property
    def force_face(self):
        """
        Face (H-rep) of the contact-force friction cone in world frame.
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
        Rays (V-rep) of the contact-force friction cone in world frame.
        """
        mu = self.static_friction / sqrt(2)  # inner approximation
        f1 = dot(self.R, [+mu, +mu, +1])
        f2 = dot(self.R, [+mu, -mu, +1])
        f3 = dot(self.R, [-mu, +mu, +1])
        f4 = dot(self.R, [-mu, -mu, +1])
        return [f1, f2, f3, f4]

    """
    Wrench Friction Cone
    ====================
    """

    @property
    def wrench_face(self):
        """
        Compute the matrix F of friction inequalities.

        This matrix describes the linearized Coulomb friction model by:

            F * w <= 0

        where w is the contact wrench at the contact point (self.p) in the
        world frame. See [Caron2015]_ for details.

        REFERENCES:

        .. [Caron2015] S. Caron, Q.-C. Pham, Y. Nakamura. "Stability of Surface
           Contacts for Humanoid Robots Closed-Form Formulae of the Contact
           Wrench Cone for Rectangular Support Areas". ICRA 2015.
           <https://scaron.info/papers/conf/caron-icra-2015.pdf>

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

            w = S * lambda,     lambda >= 0

        where S is the friction span and lambda is a vector with positive
        coordinates. Note that the contact wrench w is taken at the contact
        point (self.p) and in the world frame.
        """
        span_blocks = []
        for (i, v) in enumerate(self.vertices):
            x, y, z = v - self.p
            Gi = vstack([eye(3), crossmat(v - self.p)])
            span_blocks.append(dot(Gi, self.force_span))
        S = hstack(span_blocks)
        assert S.shape == (6, 16)
        return S
