#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2019 Stephane Caron <stephane.caron@lirmm.fr>
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
This example implements a basic stabilizer for the inverted pendulum model
based on proportional feedback of the divergent component of motion.
"""

import IPython
import numpy

from numpy import dot

import pymanoid

from pymanoid.gui import draw_arrow, draw_point
from pymanoid.misc import norm


class Stabilizer(pymanoid.Process):

    """
    Inverted pendulum stabilizer based on proportional DCM feedback.

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to stabilizer.
    gain : scalar
        DCM feedback gain.
    """

    def __init__(self, pendulum, gain=2.):
        super(Stabilizer, self).__init__()
        assert gain > 1., "DCM feedback gain needs to be greater than one"
        self.desired_dcm = pendulum.com.p
        self.desired_vrp = pendulum.com.p
        self.gain = gain
        self.handles = None
        self.omega = numpy.sqrt(pendulum.lambda_)
        self.omega2 = self.omega ** 2
        self.pendulum = pendulum

    def draw(self, dcm, cop):
        """
        Draw extra points to illustrate stabilizer behavior.

        Parameters
        ----------
        dcm : (3,) array
            Divergent component of motion.
        cop : (3,) array
            Center of pressure.
        """
        n = self.pendulum.contact.n
        dcm_ground = dcm - dot(n, dcm) * n
        self.handles = [
            draw_point(dcm_ground, color='b'),
            draw_point(cop, color='g')]

    def on_tick(self, sim):
        """
        Set inverted pendulum CoP and stiffness inputs.

        Parameters
        ----------
        sim : pymanoid.Simulation
            Simulation instance.

        Notes
        -----
        See [Englsberger2015]_ for details on the definition of the virtual
        repellent point (VRP). Here we differentiate between the constants
        lambda and omega: lambda corresponds to the "CoP-based inverted
        pendulum" and omega to the "floating-base inverted pendulum" models
        described in Section II.B of [Caron17w]_.

        Overall, we can interpret ``omega`` as a normalized stiffness between
        the CoM and VRP, while ``lambda`` corresponds to the virtual leg
        stiffness between the CoM and ZMP. The reason why the mapping is
        nonlinear is that the ZMP is constrained to lie on the contact surface,
        while the CoM can move in 3D.

        If we study further the relationship between lambda and omega, we find
        that they are actually related by a Riccati equation [Caron19]_.
        """
        com = self.pendulum.com.p
        comd = self.pendulum.com.pd
        dcm = com + comd / self.omega
        dcm_error = self.desired_dcm - dcm
        vrp = self.desired_vrp - self.gain * dcm_error
        n = self.pendulum.contact.n
        gravito_inertial_force = self.omega2 * (com - vrp) - sim.gravity
        displacement = com - self.pendulum.contact.p
        lambda_ = dot(n, gravito_inertial_force) / dot(n, displacement)
        cop = com - gravito_inertial_force / lambda_
        self.pendulum.set_cop(cop, clamp=True)
        self.pendulum.set_lambda(lambda_)
        self.draw(dcm, cop)


class Pusher(pymanoid.Process):

    """
    Send impulses to the inverted pendulum every once in a while.

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to de-stabilize.
    gain : scalar
        Magnitude of velocity jumps.

    Notes
    -----
    You know, I've seen a lot of people walkin' 'round // With tombstones in
    their eyes // But the pusher don't care // Ah, if you live or if you die
    """

    def __init__(self, pendulum, gain=0.1):
        super(Pusher, self).__init__()
        self.gain = gain
        self.handle = None
        self.nb_ticks = 0
        self.pendulum = pendulum

    def on_tick(self, sim):
        """
        Apply regular impulses to the inverted pendulum.

        Parameters
        ----------
        sim : pymanoid.Simulation
            Simulation instance.
        """
        self.nb_ticks += 1
        if self.handle is not None and self.nb_ticks % 15 == 0:
            self.handle = None
        if self.nb_ticks % 50 == 0:
            com = self.pendulum.com.p
            comd = self.pendulum.com.pd
            dv = 2. * numpy.random.random(3) - 1.
            dv[2] *= 0.5  # push is weaker in vertical direction
            dv *= self.gain / norm(dv)
            self.pendulum.com.set_vel(comd + dv)
            self.handle = draw_arrow(com - dv, com, color='b', linewidth=0.01)


if __name__ == '__main__':
    sim = pymanoid.Simulation(dt=0.03)
    sim.set_viewer()
    sim.viewer.SetCamera([
        [-0.28985337, 0.40434395, -0.86746239, 1.40434551],
        [0.95680245, 0.1009506, -0.27265003, 0.45636871],
        [-0.02267354, -0.90901867, -0.41613816, 1.15192068],
        [0., 0., 0., 1.]])

    contact = pymanoid.Contact((0.1, 0.05), pos=[0., 0., 0.])
    pendulum = pymanoid.models.InvertedPendulum(
        pos=[0., 0., 0.8], vel=numpy.zeros(3), contact=contact)
    stabilizer = Stabilizer(pendulum)
    pusher = Pusher(pendulum)

    sim.schedule(stabilizer)  # before pendulum in schedule
    sim.schedule(pendulum)
    sim.schedule_extra(pusher)
    sim.start()

    if IPython.get_ipython() is None:  # give the user a prompt
        IPython.embed()
