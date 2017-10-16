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

from numpy import array, cosh, dot, sinh, sqrt

from body import PointMass
from gui import draw_line
from misc import warn
from sim import Process, gravity


class InvertedPendulum(Process):

    """
    Inverted pendulum model.

    Parameters
    ----------
    mass : scalar
        Total robot mass.
    pos : (3,) array
        Initial position in the world frame.
    vel : (3,) array
        Initial velocity in the world frame.
    contact : pymanoid.Contact
        Contact surface specification.
    """

    def __init__(self, mass, pos, vel, contact):
        super(InvertedPendulum, self).__init__()
        com = PointMass(pos, mass, vel)
        self.com = com
        self.contact = contact
        self.cop = array([0., 0., 0.])
        self.draw_parabola = False
        self.handle = None
        self.hidden = False
        self.lambda_ = 9.81 * (com.z - contact.z)

    def copy(self):
        """
        Copy constructor.
        """
        return InvertedPendulum(
            self.com.mass, self.com.p, self.com.pd, self.contact)

    def hide(self):
        """
        Hide the pendulum in the GUI.
        """
        self.com.hide()
        if self.handle is not None:
            self.handle.Close()
        self.hidden = True

    def set_cop(self, cop):
        """
        Update the CoP location on the contact surface.

        Parameters
        ----------
        cop : (3,) array
            CoP location in the *local* inertial frame, with origin at the
            contact point and axes parallel to the world frame.
        """
        if __debug__:
            cop_check = dot(self.contact.R.T, cop)
            if abs(cop_check[0]) > 1.05 * self.contact.shape[0] \
                    or abs(cop_check[1]) > 1.05 * self.contact.shape[1]:
                warn("CoP outside of contact area")
        self.cop = cop

    def set_lambda(self, lambda_):
        """
        Update the leg stiffness coefficient.

        Parameters
        ----------
        lambda_ : scalar
            Leg stiffness coefficient (positive).
        """
        self.lambda_ = lambda_

    def integrate(self, duration):
        """
        Integrate dynamics forward for a given duration.

        Parameters
        ----------
        duration : scalar
            Duration of forward integration.
        """
        if __debug__ and abs(dot(self.contact.n, self.cop)) > 1e-10:
            self.cop = self.cop - dot(self.cop, self.contact.n) * self.contact.n
            warn("CoP was offset from contact surface")
        omega = sqrt(self.lambda_)
        p0 = self.com.p
        pd0 = self.com.pd
        ch, sh = cosh(omega * duration), sinh(omega * duration)
        vrp = self.contact.p + self.cop - gravity / self.lambda_
        p = p0 * ch + pd0 * sh / omega - vrp * (ch - 1.)
        pd = pd0 * ch + omega * (p0 - vrp) * sh
        self.com.set_pos(p)
        self.com.set_vel(pd)

    def on_tick(self, sim):
        """
        Integrate dynamics for one simulation step.

        Parameters
        ----------
        sim : pymanoid.Simulation
            Simulation instance.
        """
        self.integrate(sim.dt)
        if not self.hidden:
            self.handle = draw_line(
                self.com.p, self.contact.p + self.cop, linewidth=4, color='g')
