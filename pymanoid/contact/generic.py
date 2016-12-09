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

import numpy

from numpy import array, dot

from pymanoid.body import Box
from pymanoid.polyhedra import Cone, Cone3D
from pymanoid.sim import get_openrave_env


class Contact(Box):

    THICKNESS = 0.01

    def __init__(self, shape, pos=None, rpy=None, pose=None,
                 static_friction=None, kinetic_friction=None, visible=True,
                 name=None):
        """
        Create a new rectangular contact.

        INPUT:

        - ``shape`` -- surface dimensions (half-length, half-width) in [m]
        - ``pos`` -- contact position in world frame
        - ``rpy`` -- contact orientation in world frame
        - ``pose`` -- initial pose (supersedes pos and rpy)
        - ``static_friction`` -- static friction coefficient
        - ``kinetic_friction`` -- kinetic friction coefficient
        - ``visible`` -- initial box visibility
        - ``name`` -- (optional) name in OpenRAVE scope
        """
        X, Y = shape
        super(Contact, self).__init__(
            X, Y, Z=self.THICKNESS, pos=pos, rpy=rpy, pose=pose,
            visible=visible, dZ=-self.THICKNESS, name=name)
        self.kinetic_friction = kinetic_friction
        self.static_friction = static_friction

    def draw_force_lines(self, length=0.25):
        """
        Draw friction cones from each vertex of the surface patch.

        INPUT:

        - ``length`` -- (optional) length of friction rays in [m]

        OUTPUT:

        A list of OpenRAVE GUI handles.
        """
        env = get_openrave_env()
        handles = []
        for c in self.vertices:
            color = [0.1, 0.1, 0.1]
            color[numpy.random.randint(3)] += 0.2
            for f in self.force_rays:
                handles.append(env.drawlinelist(
                    array([c, c + length * f]),
                    linewidth=1, colors=color))
            handles.append(env.drawlinelist(
                array([c, c + length * self.n]),
                linewidth=5, colors=color))
        return handles

    def grasp_matrix(self, p):
        """
        Compute the grasp matrix from contact point ``self.p`` to a point ``p``.

        INPUT:

        - ``p`` -- point (world frame coordinates) where the wrench is taken

        OUTPUT:

        The grasp matrix G(p) converting the local contact wrench w to the
        contact wrench w(p) at another point p:

            w(p) = G(p) * w

        All wrenches are expressed with respect to the world frame.
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
    """

    @property
    def force_cone(self):
        """Contact force friction cone."""
        return Cone3D(face=self.force_face, rays=self.force_rays)

    @property
    def force_face(self):
        """Face matrix of the force friction cone."""
        raise NotImplementedError("contact mode not instantiated")

    @property
    def force_rays(self):
        """Rays of the force friction cone."""
        raise NotImplementedError("contact mode not instantiated")

    @property
    def force_span(self):
        """Span matrix of the force friction cone in world frame."""
        return array(self.force_rays).T

    """
    Wrench Friction Cone
    ====================
    """

    @property
    def wrench_cone(self):
        """Contact wrench friction cone."""
        wrench_cone = Cone(face=self.wrench_face, rays=self.wrench_rays)
        return wrench_cone

    @property
    def wrench_face(self):
        """Face matrix of the wrench friction cone."""
        raise NotImplementedError("contact mode not instantiated")

    @property
    def wrench_rays(self):
        """Rays of the wrench friction cone."""
        raise NotImplementedError("contact mode not instantiated")

    @property
    def wrench_span(self):
        """Span matrix of the wrench friction cone in world frame."""
        raise NotImplementedError("contact mode not instantiated")
