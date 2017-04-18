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

import casadi

from bisect import bisect_left
from numpy import cosh, sinh, sqrt
from time import time

from draw import draw_line, draw_point, draw_trajectory
from misc import norm
from optim import NonlinearProgram
from sim import gravity


"""
Centroidal trajectory generation for humanoid robots.
"""


class COMStepTransit(object):

    """
    Compute a COM trajectory that transits from one footstep to the next. This
    solution is applicable over arbitrary terrains.

    Implements a short-sighted optimization with a capture-point boundedness
    constraint. The capture point is defined within the floating-base inverted
    pendulum (FIP) [Caron17]_ of constant :math:`\\omega`. This model is e.g.
    used in the `nonlinear predictive controller
    <https://github.com/stephane-caron/dynamic-walking>`_ of [Caron17]_ for
    dynamic walking over rough terrains.

    Parameters
    ----------
    duration : scalar
        Duration of the transit trajectory.
    start_com : (3,) array
        Initial COM position.
    start_comd : (3,) array
        Initial COM velocity.
    foothold : Contact
        Location of the stance foot contact.
    omega : scalar
        Constant of the floating-base inverted pendulum (c.f. [Caron17]_).
    cp_target : (3,) array
        Desired terminal capture point.
    nb_steps : int
        Number of (variable-duration) discretization steps.

    Notes
    -----
    The boundedness constraint in this optimization makes up for the
    short-sighted preview [Scianca16]_, as opposed to the long-sighted approach
    to walking [Wieber06]_ where COM boundedness results from the preview of
    several future steps.
    """

    weights = {
        'match_duration': 1.,
        'capture_point': 1e-2,
        'end_com': 1e-4,
        'min_com_accel': 1e-4,
        'center_zmp': 1e-6,
    }

    dT_max = 0.1  # as low as possible
    dT_min = 0.03   # as high as possible, but caution when > sim.dt
    p_max = [+100, +100, +100]  # [m]
    p_min = [-100, -100, -100]  # [m]
    u_max = [+100, +100, +100]  # [m] / [s]^2
    u_min = [-100, -100, -100]  # [m] / [s]^2
    v_max = [+10, +10, +10]  # [m] / [s]
    v_min = [-10, -10, -10]  # [m] / [s]

    dT_init = .5 * (dT_min + dT_max)

    def __init__(self, duration, start_com, start_comd, end_com, end_comd,
                 foothold, omega2, nb_steps, nlp_options=None):
        omega = sqrt(omega2)
        self.cp_target = end_com + end_comd / omega + gravity / omega2
        self.duration = duration
        self.end_com = end_com
        self.foothold = foothold
        self.nb_steps = nb_steps
        self.nlp_options = nlp_options if nlp_options is not None else {}
        self.omega = omega
        self.omega2 = omega2
        self.start_com = start_com
        self.start_comd = start_comd
        #
        t0 = time()
        self.build()
        t1 = time()
        self.solve()
        if __debug__:
            self.print_results()
        self.build_time = t1 - t0
        self.solve_time = time() - t1
        #
        self.cum_dT = []
        t = 0.
        for k in xrange(self.nb_steps):
            t += self.dT[k]
            self.cum_dT.append(t)

    def build(self):
        self.nlp = NonlinearProgram(solver='ipopt', options=self.nlp_options)
        foothold, omega2, omega = self.foothold, self.omega2, self.omega
        cp_target = list(self.cp_target)
        end_com = list(self.end_com)
        start_com = list(self.start_com)
        start_comd = list(self.start_comd)
        p_0 = self.nlp.new_constant('p_0', 3, start_com)
        v_0 = self.nlp.new_constant('v_0', 3, start_comd)
        p_k, v_k = p_0, v_0
        T_total = 0

        for k in xrange(self.nb_steps):
            z_min = list(foothold.p - [1, 1, 1])
            z_max = list(foothold.p + [1, 1, 1])
            u_k = self.nlp.new_variable(
                'u_%d' % k, 3, init=[0, 0, 0], lb=self.u_min, ub=self.u_max)
            z_k = self.nlp.new_variable(
                'z_%d' % k, 3, init=list(foothold.p), lb=z_min, ub=z_max)
            dT_k = self.nlp.new_variable(
                'dT_%d' % k, 1, init=[self.dT_init], lb=[self.dT_min],
                ub=[self.dT_max])

            CZ_k = z_k - foothold.p
            self.nlp.add_equality_constraint(
                u_k, self.omega2 * (p_k - z_k) + gravity)
            self.nlp.extend_cost(
                self.weights['center_zmp'] * casadi.dot(CZ_k, CZ_k) * dT_k)
            self.nlp.extend_cost(
                self.weights['min_com_accel'] * casadi.dot(u_k, u_k) * dT_k)

            # Exact integration by solving the first-order ODE
            p_next = p_k + v_k / omega * casadi.sinh(omega * dT_k) \
                + u_k / omega2 * (casadi.cosh(omega * dT_k) - 1.)
            v_next = v_k * casadi.cosh(omega * dT_k) \
                + u_k / omega * casadi.sinh(omega * dT_k)
            T_total = T_total + dT_k

            self.add_com_height_constraint(p_k)
            self.add_friction_constraint(p_k, z_k)
            self.add_linear_cop_constraints(p_k, z_k)

            p_k = self.nlp.new_variable(
                'p_%d' % (k + 1), 3, init=start_com, lb=self.p_min,
                ub=self.p_max)
            v_k = self.nlp.new_variable(
                'v_%d' % (k + 1), 3, init=start_comd, lb=self.v_min,
                ub=self.v_max)
            self.nlp.add_equality_constraint(p_next, p_k)
            self.nlp.add_equality_constraint(v_next, v_k)

        p_last, v_last, z_last = p_k, v_k, z_k
        cp_last = p_last + v_last / omega + gravity / omega2
        cp_error = cp_last - cp_target
        com_error = p_last - end_com
        self.nlp.add_equality_constraint(z_last, self.cp_target)
        self.nlp.extend_cost(
            self.weights['match_duration'] * (T_total - self.duration) ** 2)
        self.nlp.extend_cost(
            self.weights['capture_point'] * casadi.dot(cp_error, cp_error))
        self.nlp.extend_cost(
            self.weights['end_com'] * casadi.dot(com_error, com_error))
        self.nlp.create_solver()

    def add_com_height_constraint(self, p, lb=0.8 - 0.2, ub=0.8 + 0.2):
        dist = casadi.dot(p - self.foothold.p, self.foothold.n)
        self.nlp.add_constraint(dist, lb=[lb], ub=[ub])

    def add_friction_constraint(self, p, z):
        mu = self.foothold.friction
        ZG = p - z
        ZG2 = casadi.dot(ZG, ZG)
        ZGn2 = casadi.dot(ZG, self.foothold.n) ** 2
        slackness = ZG2 - (1 + mu ** 2) * ZGn2
        self.nlp.add_constraint(slackness, lb=[-self.nlp.infty], ub=[0])

    def add_linear_cop_constraints(self, p, z, scaling=0.95):
        GZ = z - p
        nb_vert = len(self.foothold.vertices)
        for (i, v) in enumerate(self.foothold.vertices):
            v_next = self.foothold.vertices[(i + 1) % nb_vert]
            v = v + (1. - scaling) * (self.foothold.p - v)
            v_next = v_next + (1. - scaling) * (self.foothold.p - v_next)
            slackness = casadi.dot(v_next - v, casadi.cross(v - p, GZ))
            self.nlp.add_constraint(
                slackness, lb=[-self.nlp.infty], ub=[-0.005])

    def solve(self):
        t_solve_start = time()
        X = self.nlp.solve()
        self.solve_time = time() - t_solve_start
        Y = X[:-6].reshape((self.nb_steps, 3 + 3 + 3 + 3 + 1))
        self.P = Y[:, 0:3]
        self.V = Y[:, 3:6]
        self.Z = Y[:, 9:12]
        self.dT = Y[:, 12]
        self.p_last = X[-6:-3]
        self.v_last = X[-3:]

    def eval_state(self, t):
        omega2, omega = self.omega2, self.omega
        k = bisect_left(self.cum_dT, t)
        t0 = self.cum_dT[k]
        p0, v0, z0 = self.P[k], self.V[k], self.Z[k]
        return p0, v0, z0
        u0 = omega2 * (p0 - z0) + gravity
        dt = t - t0
        p = p0 + v0 / omega * sinh(omega * dt) \
            + u0 / omega2 * (cosh(omega * dt) - 1.)
        v = v0 * cosh(omega * dt) + u0 / omega * sinh(omega * dt)
        return p, v, z0

    def __call__(self, t):
        p, _, _ = self.eval_state(t)
        return p

    def print_results(self):
        cp_last = self.p_last + self.v_last / self.omega + gravity / self.omega2
        cp_error = norm(cp_last - self.cp_target)
        print "\n"
        print "%14s:  " % "dT's", [round(x, 3) for x in self.dT]
        print "%14s:  " % "dT_min", "%.3f s" % self.dT_min
        print "%14s:  " % "Desired TTHS", "%.3f s" % self.duration
        print "%14s:  " % "Achieved TTHS", "%.3f s" % sum(self.dT)
        print "%14s:  " % "CP error", "%.3f cm" % (100 * cp_error)
        print "%14s:  " % "Comp. time", "%.1f ms" % (1000 * self.nlp.solve_time)
        print "%14s:  " % "Iter. count", self.nlp.iter_count
        print "%14s:  " % "Status", self.nlp.return_status
        print "\n"

    def draw(self, color='b'):
        """
        Draw the COM trajectory.

        Parameters
        ----------
        color : char or triplet, optional
            Color letter or RGB values, default is 'b' for green.

        Returns
        -------
        handle : openravepy.GraphHandle
            OpenRAVE graphical handle. Must be stored in some variable,
            otherwise the drawn object will vanish instantly.
        """
        handles = draw_trajectory(self.P, color=color)
        com_target = self.cp_target - gravity / self.omega2
        com_last = self.p_last
        cp_last = self.p_last + self.v_last / self.omega + gravity / self.omega2
        handles.extend([
            draw_point(com_target, color='b', pointsize=0.025),
            draw_point(self.cp_target, color='b', pointsize=0.025),
            draw_line(com_target, self.cp_target, color='b', linewidth=1),
            draw_point(com_last, color='g', pointsize=0.025),
            draw_point(cp_last, color='g', pointsize=0.025),
            draw_line(com_last, cp_last, color='g', linewidth=1),
        ])
        return handles
