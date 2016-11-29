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

from generic import Contact

from numpy import array, dot, sqrt, zeros

from pymanoid.misc import norm
from pymanoid.sim import Process


class SlidingContact(Contact):

    def __init__(self, X, Y, pos=None, rpy=None, pose=None,
                 static_friction=None, kinetic_friction=None, visible=True,
                 name=None):
        """
        Create a new rectangular contact in sliding contact mode.

        INPUT:

        - ``X`` -- half-length of the contact surface
        - ``Y`` -- half-width of the contact surface
        - ``pos`` -- contact position in world frame
        - ``rpy`` -- contact orientation in world frame
        - ``pose`` -- initial pose (supersedes pos and rpy)
        - ``static_friction`` -- (optional) static friction coefficient
        - ``kinetic_friction`` -- kinetic friction coefficient
        - ``visible`` -- initial box visibility
        - ``name`` -- (optional) name in OpenRAVE scope
        """
        if not name:
            name = "SlidingContact-%s" % str(uuid.uuid1())[0:3]
        super(SlidingContact, self).__init__(
            X, Y, pos, rpy, pose, static_friction, kinetic_friction, visible,
            name)
        self.v = zeros(3)

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
        mu = self.friction / sqrt(2)  # inner approximation
        nv = norm(self.v)
        vx, vy, _ = self.v
        local_cone = array([
            [-1, 0, -mu * vx / nv],
            [+1, 0, +mu * vx / nv],
            [0, -1, -mu * vy / nv],
            [0, +1, -mu * vy / nv]])
        return dot(local_cone, self.R.T)

    @property
    def force_rays(self):
        """
        Rays (V-rep) of the contact-force friction cone in world frame.
        """
        mu = self.friction / sqrt(2)  # inner approximation
        nv = norm(self.v)
        vx, vy, _ = self.v
        return dot(self.R, [-mu * vx / nv, -mu * vy / nv, +1])

    """
    Wrench Friction Cone
    ====================
    """

    @property
    def wrench_face(self):
        raise NotImplementedError()

    @property
    def wrench_rays(self):
        raise NotImplementedError()

    @property
    def wrench_span(self):
        raise NotImplementedError()

    """
    Forward Dynamics
    ================
    """

    def init_fd(self):
        """Initialize forward dynamics."""
        class FDProcess(Process):
            def on_tick(_, sim):
                self.step_fd(sim.dt)

        return FDProcess()

    def step_fd(self, dt):
        self.apply_twist(self.v, zeros(3), dt)
        if self.acc_duration > 0:
            self.v += self.a * dt
            self.acc_duration -= dt
