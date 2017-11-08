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

from numpy import cosh, dot, sinh, sqrt

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
    lambda_min : scalar
        Minimum virtual leg stiffness.
    lambda_max : scalar
        Maximum virtual leg stiffness.
    visible : bool, optional
        Draw the pendulum model in GUI?
    """

    def __init__(self, mass, pos, vel, contact, lambda_min=None,
                 lambda_max=None, visible=True):
        super(InvertedPendulum, self).__init__()
        com = PointMass(pos, mass, vel)
        if not visible:
            com.hide()
        self.com = com
        self.contact = contact
        self.cop = contact.p
        self.draw_parabola = False
        self.handle = None
        self.is_visible = visible
        self.lambda_ = 9.81 * (com.z - contact.z)
        self.lambda_max = lambda_max
        self.lambda_min = lambda_min

    def copy(self, visible=True):
        """
        Copy constructor.

        Parameters
        ----------
        visible : bool, optional
            Should the copy be visible?
        """
        return InvertedPendulum(
            self.com.mass, self.com.p, self.com.pd, self.contact,
            visible=visible)

    def hide(self):
        """Hide the pendulum in the GUI."""
        self.com.hide()
        if self.handle is not None:
            self.handle.Close()
        self.is_visible = False

    def set_contact(self, contact):
        """
        Update the contact the pendulum rests upon.

        Parameters
        ----------
        contact : pymanoid.Contact
            New contact where CoPs can be realized.
        """
        self.contact = contact

    def set_cop(self, cop):
        """
        Update the CoP location on the contact surface.

        Parameters
        ----------
        cop : (3,) array
            New CoP location in the world frame.
        """
        if __debug__:
            cop_check = dot(self.contact.R.T, cop - self.contact.p)
            if abs(cop_check[0]) > 1.05 * self.contact.shape[0]:
                warn("CoP crosses contact area along sagittal axis")
            if abs(cop_check[1]) > 1.05 * self.contact.shape[1]:
                warn("CoP crosses contact area along lateral axis")
            if abs(cop_check[2]) > 0.05:
                warn("CoP does not lie on contact area")
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
        omega = sqrt(self.lambda_)
        p0 = self.com.p
        pd0 = self.com.pd
        ch, sh = cosh(omega * duration), sinh(omega * duration)
        vrp = self.cop - gravity / self.lambda_
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
        if self.is_visible:
            self.handle = draw_line(
                self.com.p, self.cop, linewidth=4, color='g')
