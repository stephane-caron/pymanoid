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
import uuid

from numpy import array, dot, zeros

from pymanoid.body import Box
from pymanoid.polyhedra import Cone, Cone3D
from pymanoid.sim import get_openrave_env


class Contact(Box):

    THICKNESS = 0.01

    def __init__(self, X, Y, pos=None, rpy=None, friction=None,
                 pose=None, visible=True, name=None):
        """
        Create a new rectangular contact.

        INPUT:

        - ``X`` -- half-length of the contact surface
        - ``Y`` -- half-width of the contact surface
        - ``pos`` -- contact position in world frame
        - ``rpy`` -- contact orientation in world frame
        - ``friction`` -- friction coefficient
        - ``pose`` -- initial pose (supersedes pos and rpy)
        - ``visible`` -- initial box visibility
        """
        if not name:
            name = "Contact-%s" % str(uuid.uuid1())[0:3]
        self.friction = friction
        self.v = zeros(3)
        super(Contact, self).__init__(
            X, Y, Z=self.THICKNESS, pos=pos, rpy=rpy, pose=pose,
            visible=visible, dZ=-self.THICKNESS, name=name)

    @property
    def dict_repr(self):
        return {
            'X': self.X,
            'Y': self.Y,
            'Z': self.Z,
            'pos': list(self.p),
            'rpy': list(self.rpy),
            'friction': self.friction,
            'visible': self.is_visible,
        }

    def draw_force_lines(self, length=0.25):
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

    @property
    def force_cone(self):
        """
        Contact-force friction cone.
        """
        return Cone3D(face=self.force_face, rays=self.force_rays)

    @property
    def force_face(self):
        raise NotImplementedError("contact mode not instantiated")

    @property
    def force_rays(self):
        raise NotImplementedError("contact mode not instantiated")

    @property
    def force_span(self):
        """
        Span matrix of the contact-force friction cone in world frame.
        """
        return array(self.force_rays).T

    def grasp_matrix(self, p):
        """
        Compute the grasp matrix from contact point to ``p`` in the world frame.

        INPUT:

        - ``p`` -- end point where the resultant wrench is taken

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

    @property
    def wrench_cone(self):
        """
        Contact-wrench friction cone.
        """
        wrench_cone = Cone(face=self.wrench_face, rays=self.wrench_rays)
        return wrench_cone

    @property
    def wrench_face(self):
        raise NotImplementedError("contact mode not instantiated")

    @property
    def wrench_rays(self):
        raise NotImplementedError("contact mode not instantiated")

    @property
    def wrench_span(self):
        raise NotImplementedError("contact mode not instantiated")
