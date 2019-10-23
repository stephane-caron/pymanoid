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

try:
    import casadi
except ImportError:
    casadi = None

from bisect import bisect_left
from numpy import cosh, sinh, sqrt
from time import time

from .gui import draw_line, draw_point, draw_trajectory
from .misc import norm
from .nlp import NonlinearProgram
from .sim import gravity


"""
Centroidal trajectory generation for humanoid robots.
"""


class COMStepTransit(object):

    """
    Compute a COM trajectory that transits from one footstep to the next. This
    solution is applicable over arbitrary terrains.

    Implements a short-sighted optimization with a DCM boundedness constraint,
    defined within the floating-base inverted pendulum (FIP) of constant
    :math:`\\omega`. This approach is used in the `nonlinear predictive
    controller <https://github.com/stephane-caron/fip-walkgen>`_ of [Caron17w]_
    for dynamic walking over rough terrains.

    Parameters
    ----------
    desired_duration : scalar
        Desired duration of the transit trajectory.
    start_com : (3,) array
        Initial COM position.
    start_comd : (3,) array
        Initial COM velocity.
    foothold : Contact
        Location of the stance foot contact.
    omega : scalar
        Constant of the floating-base inverted pendulum.
    dcm_target : (3,) array
        Desired terminal divergent-component of COM motion.
    nb_steps : int
        Number of discretization steps.

    Notes
    -----
    The boundedness constraint in this optimization makes up for the
    short-sighted preview [Scianca16]_, as opposed to the long-sighted approach
    to walking [Wieber06]_ where COM boundedness results from the preview of
    several future steps.

    The benefit of using ZMP controls rather than COM accelerations lies in the
    exact integration scheme. This phenomenon appears in flat-floor models as
    well: in the CART-table model, segments with fixed COM acceleration imply
    variable ZMPs that may exit the support area (see e.g. Figure 3.17 in
    [ElKhoury13]_).
    """

    weights = {
        'match_dcm': 1.,
        'match_duration': 1e-2,
        'center_zmp': 1e-4,
        'minimize_comdd': [1e-6, 1e-6, 1e-4],  # x-y-z weights
    }

    dT_min = 0.03                 # [s], caution when > sim.dt
    p_max = [+100, +100, +100]    # [m]
    p_min = [-100, -100, -100]    # [m]
    pd_max = [+10, +10, +10]      # [m] / [s]
    pd_min = [-10, -10, -10]      # [m] / [s]
    pdd_max = [+100, +100, +100]  # [m] / [s]^2
    pdd_min = [-100, -100, -100]  # [m] / [s]^2

    def __init__(self, desired_duration, start_com, start_comd, dcm_target,
                 foothold, next_foothold, omega2, nb_steps, nlp_options=None):
        assert casadi is not None, "CasADi is not installed"
        self.dT = None
        self.dcm_target = dcm_target
        self.desired_duration = desired_duration
        self.duration = None
        self.foothold = foothold
        self.nb_steps = nb_steps
        self.next_foothold = next_foothold
        self.nlp_options = nlp_options if nlp_options is not None else {}
        self.omega = sqrt(omega2)
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
        for k in range(self.nb_steps):
            t += self.dT
            self.cum_dT.append(t)

    def build(self):
        """
        Build the internal nonlinear program (NLP).
        """
        self.nlp = NonlinearProgram(solver='ipopt', options=self.nlp_options)
        foothold, next_foothold = self.foothold, self.next_foothold
        omega2, omega = self.omega2, self.omega
        dcm_target = list(self.dcm_target)
        start_com = list(self.start_com)
        start_comd = list(self.start_comd)
        T = self.nlp.new_variable(
            'T', 1, init=[self.desired_duration],
            lb=[self.dT_min * self.nb_steps], ub=[5. * self.desired_duration])
        p_0 = self.nlp.new_constant('p_0', 3, start_com)
        pd_0 = self.nlp.new_constant('pd_0', 3, start_comd)
        dT = T / self.nb_steps
        p_k, pd_k = p_0, pd_0
        W_comdd = list(self.weights['minimize_comdd'])
        assert len(W_comdd) in [1, 3]

        for k in range(self.nb_steps):
            z_min = list(foothold.p - [0.42, 0.42, 0.42])
            z_max = list(foothold.p + [0.42, 0.42, 0.42])
            pdd_k = self.nlp.new_variable(
                'pdd_%d' % k, 3, init=[0, 0, 0], lb=self.pdd_min,
                ub=self.pdd_max)
            z_k = self.nlp.new_variable(
                'z_%d' % k, 3, init=list(foothold.p), lb=z_min, ub=z_max)

            CZ_k = z_k - foothold.p
            self.nlp.add_equality_constraint(
                pdd_k, self.omega2 * (p_k - z_k) + gravity)
            self.nlp.extend_cost(
                self.weights['center_zmp'] * casadi.dot(CZ_k, CZ_k) * dT)
            self.nlp.extend_cost(
                casadi.dot(pdd_k, W_comdd * pdd_k) * dT)

            # Exact integration by solving the first-order ODE
            p_next = p_k + pd_k / omega * casadi.sinh(omega * dT) \
                + pdd_k / omega2 * (casadi.cosh(omega * dT) - 1.)
            pd_next = pd_k * casadi.cosh(omega * dT) \
                + pdd_k / omega * casadi.sinh(omega * dT)

            self.add_com_height_constraint(p_k, ref_height=0.8, max_dev=0.2)
            self.add_friction_constraint(p_k, z_k, foothold)
            self.add_linear_cop_constraints(p_k, z_k, foothold)

            p_k = self.nlp.new_variable(
                'p_%d' % (k + 1), 3, init=start_com, lb=self.p_min,
                ub=self.p_max)
            pd_k = self.nlp.new_variable(
                'pd_%d' % (k + 1), 3, init=start_comd, lb=self.pd_min,
                ub=self.pd_max)
            self.nlp.add_equality_constraint(p_next, p_k)
            self.nlp.add_equality_constraint(pd_next, pd_k)

        p_last, pd_last = p_k, pd_k
        dcm_last = p_last + pd_last / omega
        cp_last = dcm_last + gravity / omega2
        self.add_friction_constraint(p_last, cp_last, next_foothold)
        self.add_linear_cop_constraints(p_last, cp_last, next_foothold)

        dcm_error = dcm_last - dcm_target
        duration_error = T - self.desired_duration
        self.nlp.extend_cost(
            self.weights['match_dcm'] * casadi.dot(dcm_error, dcm_error))
        self.nlp.extend_cost(
            self.weights['match_duration'] * duration_error ** 2)
        self.nlp.create_solver()

    def add_com_height_constraint(self, p, ref_height, max_dev):
        """
        Constraint the COM to deviate by at most `max_dev` from `ref_height`.

        Parameters
        ----------
        ref_height : scalar
            Reference COM height.
        max_dev : scalar
            Maximum distance allowed from the reference.

        Note
        ----
        The height is measured along the z-axis of the contact frame here, not
        of the world frame's.
        """
        lb, ub = ref_height - max_dev, ref_height + max_dev
        dist = casadi.dot(p - self.foothold.p, self.foothold.n)
        self.nlp.add_constraint(dist, lb=[lb], ub=[ub])

    def add_friction_constraint(self, p, z, foothold):
        """
        Add a circular friction cone constraint for a COM located at `p` and a
        (floating-base) ZMP located at `z`.

        Parameters
        ----------
        p : casadi.MX
            Symbol of COM position variable.
        z : casadi.MX
            Symbol of ZMP position variable.
        """
        mu = foothold.friction
        ZG = p - z
        ZG2 = casadi.dot(ZG, ZG)
        ZGn2 = casadi.dot(ZG, foothold.n) ** 2
        slackness = ZG2 - (1 + mu ** 2) * ZGn2
        self.nlp.add_constraint(slackness, lb=[-self.nlp.infty], ub=[-0.005])

    def add_linear_cop_constraints(self, p, z, foothold, scaling=0.95):
        """
        Constraint the COP, located between the COM `p` and the (floating-base)
        ZMP `z`, to lie inside the contact area.

        Parameters
        ----------
        p : casadi.MX
            Symbol of COM position variable.
        z : casadi.MX
            Symbol of ZMP position variable.
        scaling : scalar, optional
            Scaling factor between 0 and 1 applied to the contact area.
        """
        GZ = z - p
        nb_vert = len(foothold.vertices)
        for (i, v) in enumerate(foothold.vertices):
            v_next = foothold.vertices[(i + 1) % nb_vert]
            v = v + (1. - scaling) * (foothold.p - v)
            v_next = v_next + (1. - scaling) * (foothold.p - v_next)
            slackness = casadi.dot(v_next - v, casadi.cross(v - p, GZ))
            self.nlp.add_constraint(
                slackness, lb=[-self.nlp.infty], ub=[-0.0005])

    def solve(self):
        """
        Solve the nonlinear program and store the solution, if found.
        """
        t_solve_start = time()
        X = self.nlp.solve()
        t_solve_end = time()
        Y = X[1:-6].reshape((self.nb_steps, 3 + 3 + 3 + 3))
        duration = X[0]
        p_last = X[-6:-3]
        pd_last = X[-3:]
        self.P = Y[:, 0:3]
        self.U = Y[:, 6:9]
        self.V = Y[:, 3:6]
        self.Z = Y[:, 9:12]
        self.cp_last = p_last + pd_last / self.omega + gravity / self.omega2
        self.dT = duration / self.nb_steps
        self.duration = duration
        self.p_last = p_last
        self.pd_last = pd_last
        self.solve_time = t_solve_end - t_solve_start

    def __call__(self, t, field=None):
        """
        Evaluate the solution's state at time `t`.

        Parameters
        ----------
        t : scalar
            Time over the solution trajectory.
        field : string, optional
            Field to return: 'p' for position, 'pd' for velocity, 'pdd' for
            acceleration or 'z' for floating-base ZMP.

        Returns
        -------
        value : (3,) array or array tuple
            Value of the requested field at time `t`, if a request there was.
            Otherwise, tuple (`p`, `pd`, `pdd`, `z`) with all fields.

        Notes
        -----
        This function can also be used to read past the end of the preview
        horizon. In this case, the post-horizon strategy is applied: ZMP fixed
        at the capture point, so that the COM will converge to the DCM target.
        """
        k = bisect_left(self.cum_dT, t)
        if k < self.nb_steps:
            p0, pd0, z = self.P[k], self.V[k], self.Z[k]
        else:  # t > duration of the trajectory
            p0, pd0, z = self.p_last, self.pd_last, self.cp_last
        if field == 'z':
            return z
        omega2 = self.omega2
        pdd0 = omega2 * (p0 - z) + gravity
        omega = self.omega
        t0 = self.cum_dT[k - 1] if k > 0 else 0.
        dt = t - t0
        pd = pd0 * cosh(omega * dt) + pdd0 / omega * sinh(omega * dt)
        if field == 'pd':
            return pd
        p = p0 + pd0 / omega * sinh(omega * dt) \
            + pdd0 / omega2 * (cosh(omega * dt) - 1.)
        if field == 'p':
            return p
        pdd = omega2 * (p - z) + gravity
        return (p, pd, pdd, z)

    def print_results(self):
        """
        Print various statistics on NLP resolution.
        """
        dcm_last = self.p_last + self.pd_last / self.omega
        dcm_error = norm(dcm_last - self.dcm_target)
        comp_time_s = self.nlp.solve_time
        print("\n")
        print("%14s: " % "Desired dur.", "%.3f s" % self.desired_duration)
        print("%14s: " % "Duration", "%.3f s" % self.duration)
        print("%14s: " % "DCM error", "%.3f cm" % (100 * dcm_error))
        print("%14s: " % "Comp. time", "%.1f ms" % (1000 * comp_time_s))
        print("%14s: " % "Iter. count", self.nlp.iter_count)
        print("%14s: " % "Status", self.nlp.return_status)
        print("\n")

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
        handles.extend(draw_trajectory(self.Z, color='m'))
        com_last = self.p_last
        dcm_last = self.p_last + self.pd_last / self.omega
        cp_target = self.dcm_target + gravity / self.omega2
        cp_last = dcm_last + gravity / self.omega2
        for k in range(self.nb_steps):
            p, z = self.P[k], self.Z[k]
            handles.append(draw_line(z, p, color='c', linewidth=0.2))
        handles.extend([
            draw_point(self.dcm_target, color='b', pointsize=0.025),
            draw_point(cp_target, color='b', pointsize=0.025),
            draw_line(self.dcm_target, cp_target, color='b', linewidth=1),
            draw_point(com_last, color='g', pointsize=0.025),
            draw_point(cp_last, color='g', pointsize=0.025),
            draw_line(com_last, cp_last, color='g', linewidth=1),
        ])
        return handles
