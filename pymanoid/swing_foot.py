#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2017 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under
# the terms of the GNU Lesser General Public License as published by the Free
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

from numpy import array, dot, eye, hstack, linspace, zeros
from openravepy import InterpolateQuatSlerp as quat_slerp
from pymanoid.gui import draw_trajectory
from pymanoid.interp import interpolate_cubic_hermite
from qpsolvers import solve_qp


def factor_cubic_hermite_curve(p0, n0, p1, n1):
    """
    Let `H` denote the Hermite curve (parameterized by `\\lambda` and `\\mu`)
    such that:

    .. math::

        \\begin{array}{rcl}
            H(0) & = & p_0 \\\\
            H'(0) & = & \\lambda n_0 \\\\
            H(1) & = & p_1 \\\\
            H'(1) & = & \\mu n_1 \\\\
        \\end{array}

    This function returns the three factors of `H` corresponding to the symbols
    :math:`\\lambda`, :math:`\\mu` and :math:`1` (constant part).

    Parameters
    ----------
    p0 : (3,) array
        Initial position.
    n0 : (3,) array
        Initial tangent vector.
    p1 : (3,) array
        Final position.
    n1 : (3,) array
        Final tangent vector.

    Returns
    -------
    H_lambda : function
        Factor of :math:`\\lambda` in `H`.
    H_mu : function
        Factor of :math:`\\mu` in `H`.
    H_cst : function
        Part of `H` that depends neither on :math:`\\lambda` nor :math:`\\mu`.
    """
    def H_lambda(s):
        return s * (1 + s * (s - 2)) * n0

    def H_mu(s):
        return s ** 2 * (s - 1) * n1

    def H_cst(s):
        return p0 + s ** 2 * (3 - 2 * s) * (p1 - p0)

    return H_lambda, H_mu, H_cst


class SwingFoot(object):

    """
    Polynomial swing foot interpolator.

    Parameters
    ----------
    start_contact : pymanoid.Contact
        Initial contact.
    end_contact : pymanoid.Contact
        Target contact.
    duration : scalar
        Swing duration in [s].
    """

    default_start_clearance = 0.05  # [m]
    default_end_clearance = 0.1  # [m]

    def __init__(self, start_contact, end_contact, duration):
        self.draw_trajectory = False
        self.duration = duration
        self.end_contact = end_contact.copy(hide=True)
        self.foot_vel = zeros(3)
        self.playback_time = 0.
        self.start_contact = start_contact.copy(hide=True)
        #
        self.path = self.interpolate()

    def interpolate(self):
        """
        Interpolate swing foot path.

        Returns
        -------
        path : pymanoid.NDPolynomial
            Polynomial path with index between 0 and 1.
        """
        n0 = self.start_contact.n
        n1 = self.end_contact.n
        p0 = self.start_contact.p
        p1 = self.end_contact.p
        start_clearance = self.default_start_clearance
        end_clearance = self.default_end_clearance
        if hasattr(self.end_contact, 'landing_clearance'):
            end_clearance = self.end_contact.landing_clearance
        if hasattr(self.end_contact, 'landing_tangent'):
            n1 = self.end_contact.landing_tangent
        if hasattr(self.start_contact, 'takeoff_clearance'):
            start_clearance = self.start_contact.takeoff_clearance
        if hasattr(self.start_contact, 'takeoff_tangent'):
            n0 = self.start_contact.takeoff_tangent
        # H(s) = H_lambda(s) * lambda + H_mu(s) * mu + H_cst(s)
        H_lambda, H_mu, H_cst = factor_cubic_hermite_curve(p0, n0, p1, n1)
        s0 = 1. / 4
        a0 = dot(H_lambda(s0), n0)
        b0 = dot(H_mu(s0), n0)
        c0 = dot(H_cst(s0) - p0, n0)
        h0 = start_clearance
        # a0 * lambda + b0 * mu + c0 >= h0
        s1 = 3. / 4
        a1 = dot(H_lambda(s1), n1)
        b1 = dot(H_mu(s1), n1)
        c1 = dot(H_cst(s1) - p1, n1)
        h1 = end_clearance
        # a1 * lambda + b1 * mu + c1 >= h1
        P = eye(2)
        q = zeros(2)
        G = array([[-a0, -b0], [-a1, -b1]])
        h = array([c0 - h0, c1 - h1])
        x = solve_qp(P, q, G, h)
        # H = lambda s: H_lambda(s) * x[0] + H_mu(s) * x[1] + H_cst(s)
        path = interpolate_cubic_hermite(p0, x[0] * n0, p1, x[1] * n1)
        return path

    def draw(self, color='r'):
        """
        Draw swing foot trajectory.

        Parameters
        ----------
        color : char or triplet, optional
            Color letter or RGB values, default is 'b' for blue.
        """
        points = [self.path(s) for s in linspace(0, 1, 10)]
        return draw_trajectory(points, color=color)

    def integrate(self, dt):
        """
        Integrate swing foot motion forward by a given amount of time.

        Parameters
        ----------
        dt : scalar
            Duration of forward integration, in [s].
        """
        self.playback_time += dt
        s = min(1., self.playback_time / self.duration)
        quat = quat_slerp(self.start_contact.quat, self.end_contact.quat, s)
        pos = self.path(s)
        pose = hstack([quat, pos])
        return pose
