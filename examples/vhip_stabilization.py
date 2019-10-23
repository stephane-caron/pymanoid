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
This script comes with the research paper "Biped Stabilization by Linear
Feedback of the Variable-Height Inverted Pendulum Model" (Caron, 2019)
<https://hal.archives-ouvertes.fr/hal-02289919v1/document>.

In this example, we compare two stabilizers for the inverted pendulum model.
The first one (baseline) is based on proportional feedback of the 3D DCM
<https://doi.org/10.1109/TRO.2015.2405592>. The second one (proposed) performs
proportional feedback of a 4D DCM of the same model; see the paper for details.
"""

import IPython
import numpy
import scipy.signal
import sys

try:
    import cvxpy
except ImportError:
    raise ImportError("This example requires CVXPY, install it e.g. via pip")

from numpy import array, dot, eye, hstack, sqrt, vstack, zeros
from qpsolvers import solve_qp

import pymanoid

from pymanoid.sim import gravity


mass = 38.  # [kg]
max_dcm_height = 1.  # [m]
max_force = 1000.  # [N]
min_dcm_height = 0.5  # [m]
min_force = 1.  # [N]
ref_offset = array([0.0, 0.0, 0.])  # [m]
k_p = 3.  # proportional DCM feedback gain

assert k_p > 1., "DCM feedback gain needs to be greater than one"


class Stabilizer(pymanoid.Process):

    """
    Base class for stabilizer processes.

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to stabilize.

    Attributes
    ----------
    contact : pymanoid.Contact
        Contact frame and area dimensions.
    dcm : (3,) array
        Position of the DCM in the world frame.
    omega : scalar
        Instantaneous natural frequency of the pendulum.
    pendulum : pymanoid.InvertedPendulum
        Measured state of the reduced model.
    ref_com : (3,) array
        Desired center of mass (CoM) position.
    ref_comd : (3,) array
        Desired CoM velocity.
    ref_cop : (3,) array
        Desired center of pressure (CoP).
    ref_lambda : scalar
        Desired normalized leg stiffness.
    ref_omega : scalar
        Desired natural frequency.
    """

    def __init__(self, pendulum):
        super(Stabilizer, self).__init__()
        ref_com = pendulum.com.p + ref_offset
        n = pendulum.contact.normal
        lambda_ = -dot(n, gravity) / dot(n, ref_com - pendulum.contact.p)
        omega = sqrt(lambda_)
        ref_cop = ref_com + gravity / lambda_
        assert abs(lambda_ - pendulum.lambda_) < 1e-5
        self.contact = pendulum.contact
        self.dcm = ref_com
        self.omega = omega
        self.pendulum = pendulum
        self.ref_com = ref_com
        self.ref_comd = numpy.zeros(3)
        self.ref_cop = ref_cop
        self.ref_lambda = lambda_
        self.ref_omega = omega

    def reset_pendulum(self):
        """
        Reset inverted pendulum to its reference state.
        """
        self.omega = self.ref_omega
        self.pendulum.com.set_pos(self.ref_com)
        self.pendulum.com.set_vel(self.ref_comd)
        self.pendulum.set_cop(self.ref_cop)
        self.pendulum.set_lambda(self.ref_lambda)

    def on_tick(self, sim):
        """
        Set inverted pendulum CoP and stiffness inputs.

        Parameters
        ----------
        sim : pymanoid.Simulation
            Simulation instance.
        """
        Delta_r, Delta_lambda = self.compute_compensation()
        cop = self.ref_cop + dot(self.contact.R[:3, :2], Delta_r)
        lambda_ = self.ref_lambda + Delta_lambda
        self.pendulum.set_cop(cop)
        self.pendulum.set_lambda(lambda_)


class VRPStabilizer(Stabilizer):

    """
    Inverted pendulum stabilizer based on proportional feedback of the
    3D divergent component of motion (DCM) applied to the virtual repellent
    point (VRP).

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to stabilize.

    Attributes
    ----------
    ref_dcm : (3,) array
        Desired (3D) divergent component of motion.
    ref_vrp : (3,) array
        Desired virtual repellent point (VRP).

    Notes
    -----
    See "Three-Dimensional Bipedal Walking Control Based on Divergent Component
    of Motion" (Englsberger et al., IEEE Transactions on Robotics) for details.
    """

    def __init__(self, pendulum):
        super(VRPStabilizer, self).__init__(pendulum)
        self.ref_dcm = self.ref_com
        self.ref_vrp = self.ref_com

    def compute_compensation(self):
        """
        Compute CoP and normalized leg stiffness compensation.
        """
        omega = self.omega
        com = self.pendulum.com.p
        comd = self.pendulum.com.pd
        dcm = com + comd / omega
        Delta_dcm = dcm - self.ref_dcm
        vrp = self.ref_vrp + k_p * Delta_dcm
        n = self.pendulum.contact.n
        gravito_inertial_force = omega ** 2 * (com - vrp) - gravity
        displacement = com - self.pendulum.contact.p
        lambda_ = dot(n, gravito_inertial_force) / dot(n, displacement)
        cop = com - gravito_inertial_force / lambda_
        Delta_r = dot(self.contact.R.T, cop - self.ref_cop)[:2]
        Delta_lambda = lambda_ - self.ref_lambda
        self.dcm = dcm
        return (Delta_r, Delta_lambda)


class VHIPStabilizer(Stabilizer):

    """
    Stabilizer based on proportional feedback of the 4D divergent component of
    motion of the variable-height inverted pendulum (VHIP).

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to stabilize.

    Notes
    -----
    This implementation uses CVXPY <https://www.cvxpy.org/>. Using this
    modeling language here allowed us to try various formulations of the
    controller before converging on this one. We can only praise the agility of
    this approach, as opposed to e.g. writing QP matrices directly.

    See "Biped Stabilization by Linear Feedback of the Variable-Height Inverted
    Pendulum Model" (Caron, 2019) for detail on the controller itself.
    """

    def __init__(self, pendulum):
        super(VHIPStabilizer, self).__init__(pendulum)
        r_d_contact = dot(self.contact.R.T, self.ref_cop - self.contact.p)[:2]
        self.r_contact_max = array(self.contact.shape)
        self.ref_cop_contact = r_d_contact
        self.ref_dcm = self.ref_com
        self.ref_vrp = self.ref_com

    def compute_compensation(self):
        """
        Compute CoP and normalized leg stiffness compensation.
        """
        Delta_com = self.pendulum.com.p - self.ref_com
        Delta_comd = self.pendulum.com.pd - self.ref_comd
        measured_comd = self.pendulum.com.pd
        lambda_d = self.ref_lambda
        nu_d = self.ref_vrp
        omega_d = self.ref_omega
        r_d_contact = self.ref_cop_contact
        xi_d = self.ref_dcm
        height = dot(self.contact.normal, self.pendulum.com.p - self.contact.p)
        lambda_max = max_force / (mass * height)
        lambda_min = min_force / (mass * height)
        omega_max = sqrt(lambda_max)
        omega_min = sqrt(lambda_min)

        Delta_lambda = cvxpy.Variable(1)
        Delta_nu = cvxpy.Variable(3)
        Delta_omega = cvxpy.Variable(1)
        Delta_r = cvxpy.Variable(2)
        u = cvxpy.Variable(3)

        Delta_xi = Delta_com + Delta_comd / omega_d \
            - measured_comd / (omega_d ** 2) * Delta_omega
        Delta_omegad = 2 * omega_d * Delta_omega - Delta_lambda
        Delta_r_world = contact.R[:3, :2] * Delta_r
        r_contact = r_d_contact + Delta_r
        lambda_ = lambda_d + Delta_lambda
        omega = omega_d + Delta_omega

        Delta_xid = (
            Delta_lambda * (xi_d - nu_d)
            + lambda_d * (Delta_xi - Delta_nu) +
            - Delta_omega * lambda_d * (xi_d - nu_d) / omega_d) / omega_d
        xi_z = self.ref_dcm[2] + Delta_xi[2] + 1.5 * sim.dt * Delta_xid[2]
        costs = []
        sq_costs = [
            (1., u[0]),
            (1., u[1]),
            (1e-3, u[2])]
        for weight, expr in sq_costs:
            costs.append((weight, cvxpy.sum_squares(expr)))
        cost = sum(weight * expr for (weight, expr) in costs)
        prob = cvxpy.Problem(
            objective=cvxpy.Minimize(cost),
            constraints=[
                Delta_xid == lambda_d / omega_d * ((1 - k_p) * Delta_xi + u),
                Delta_omegad == omega_d * (1 - k_p) * Delta_omega,
                Delta_nu == Delta_r_world
                + gravity * Delta_lambda / lambda_d ** 2,
                cvxpy.abs(r_contact) <= self.r_contact_max,
                lambda_ <= lambda_max,
                lambda_ >= lambda_min,
                xi_z <= max_dcm_height,
                xi_z >= min_dcm_height,
                omega <= omega_max,
                omega >= omega_min])
        prob.solve()
        Delta_lambda_opt = Delta_lambda.value
        Delta_r_opt = array(Delta_r.value).reshape((2,))
        self.omega = omega_d + Delta_omega.value
        self.dcm = self.pendulum.com.p \
            + self.pendulum.com.pd / self.omega
        return (Delta_r_opt, Delta_lambda_opt)


class VHIPQPStabilizer(VHIPStabilizer):

    """
    Stabilizer based on proportional feedback of the 4D divergent component of
    motion of the variable-height inverted pendulum (VHIP).

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to stabilize.

    Notes
    -----
    This implementation transcripts QP matrices from :class:`VHIPStabilizer`.
    We checked that the two produce the same outputs before switching to C++ in
    <https://github.com/stephane-caron/vhip_walking_controller/>. (This step
    would not have been necessary if we had a modeling language for convex
    optimization directly in C++.)
    """

    def compute_compensation(self):
        """
        Compute CoP and normalized leg stiffness compensation.
        """
        Delta_com = self.pendulum.com.p - self.ref_com
        Delta_comd = self.pendulum.com.pd - self.ref_comd
        measured_comd = self.pendulum.com.pd
        lambda_d = self.ref_lambda
        nu_d = self.ref_vrp
        omega_d = self.ref_omega
        r_d = self.ref_cop
        r_d_contact = self.ref_cop_contact
        xi_d = self.ref_dcm
        height = dot(self.contact.normal, self.pendulum.com.p - self.contact.p)
        lambda_max = max_force / (mass * height)
        lambda_min = min_force / (mass * height)
        omega_max = sqrt(lambda_max)
        omega_min = sqrt(lambda_min)

        A = vstack([
            hstack([-k_p * eye(3),
                    (xi_d - nu_d).reshape((3, 1)) / omega_d,
                    self.contact.R[:3, :2],
                    (r_d - xi_d).reshape((3, 1)) / lambda_d,
                    eye(3)]),
            hstack([eye(3),
                    measured_comd.reshape((3, 1)) / omega_d ** 2,
                    zeros((3, 2)),
                    zeros((3, 1)),
                    zeros((3, 3))]),
            hstack([zeros((1, 3)),
                    omega_d * (1 + k_p) * eye(1),
                    zeros((1, 2)),
                    -1 * eye(1),
                    zeros((1, 3))])])
        b = hstack([
            zeros(3),
            Delta_com + Delta_comd / omega_d,
            zeros(1)])

        G_cop = array([
            [0., 0., 0., 0., +1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., -1., 0., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., +1., 0., 0., 0., 0.],
            [0., 0., 0., 0., 0., -1., 0., 0., 0., 0.]])
        h_cop = array([
            self.contact.shape[0] - r_d_contact[0],
            self.contact.shape[0] + r_d_contact[0],
            self.contact.shape[1] - r_d_contact[1],
            self.contact.shape[1] + r_d_contact[1]])

        G_lambda = array([
            [0., 0., 0., 0., 0., 0., +1., 0., 0., 0.],
            [0., 0., 0., 0., 0., 0., -1., 0., 0., 0.]])
        h_lambda = array([
            lambda_max - lambda_d,
            lambda_d - lambda_min])

        G_omega = array([
            [0., 0., 0., +1., 0., 0., 0., 0., 0., 0.],
            [0., 0., 0., -1., 0., 0., 0., 0., 0., 0.]])
        h_omega = array([
            omega_max - omega_d,
            omega_d - omega_min])

        g_sigma = 1.5 * lambda_d * sim.dt / omega_d
        g_xi = 1 + g_sigma * (1 - k_p)
        G_xi_next = array([
            [0., 0., +g_xi, 0., 0., 0., 0., 0., 0., +g_sigma],
            [0., 0., -g_xi, 0., 0., 0., 0., 0., 0., -g_sigma]])
        h_xi_next = array([
            max_dcm_height - self.ref_dcm[2],
            self.ref_dcm[2] - min_dcm_height])

        G = vstack([G_cop, G_lambda, G_omega, G_xi_next])
        h = hstack([h_cop, h_lambda, h_omega, h_xi_next])

        P = numpy.diag([1e-6] * 7 + [1., 1., 1e-3])
        q = numpy.zeros(10)

        Delta_x = solve_qp(P, q, G, h, A, b, solver='quadprog')
        Delta_omega_opt = Delta_x[3]
        Delta_r_opt = Delta_x[4:6]
        Delta_lambda_opt = Delta_x[6]
        self.omega = omega_d + Delta_omega_opt
        self.dcm = self.pendulum.com.p \
            + self.pendulum.com.pd / self.omega
        return (Delta_r_opt, Delta_lambda_opt)


class BonusPolePlacementStabilizer(Stabilizer):

    """
    This is a "bonus" stabilizer, not reported in the paper, that was an
    intermediate step in our derivation of the VHIPQPStabilizer.

    Parameters
    ----------
    pendulum : pymanoid.models.InvertedPendulum
        Inverted pendulum to stabilize.
    k_z : scalar
        Feedback gain between DCM altitude and normalized leg stiffness input.

    Notes
    -----
    This stabilizer also performs pole placement on a 4D DCM (using a velocity
    rather than position DCM though), but contrary to VHIPQPStabilizer it
    doesn't force the closed-loop matrix to be diagonal. We started out
    exploring this stabilizer first.

    The first thing to observe by direct pole placement is that the gain matrix
    has essentially four non-zero gains in general. You can try out the
    :func:`set_poles` function to verify this.

    The closed-loop system with four gains has structure: in the horizontal
    plane it is equivalent to the VRPStabilizer, and the normalized leg
    stiffness lambda depends on both the vertical DCM and the natural frequency
    omega. We observed that this system performs identically to the previous
    one in the horizontal plane, and always worse than the previous one
    vertically.

    However, raising the k_z (vertical DCM to lambda) gain to large values, we
    noticed that the vertical tracking of this stabilizer converged to that of
    the VRPStabilizer. In the limit where k_z goes to infinity, the system
    slides on the constraint given by Equation (21) in the paper. This is how
    we came to the derivation of the VHIPQPStabilizer.
    """

    def __init__(self, pendulum, k_z):
        super(BonusPolePlacementStabilizer, self).__init__(pendulum)
        ref_dcm = self.ref_comd + self.ref_omega * self.ref_com
        # ref_cop = numpy.zeros(3)  # assumption of this stabilizer
        assert numpy.linalg.norm(self.contact.R - numpy.eye(3)) < 1e-5
        A = array([
            [self.ref_omega, 0., 0., ref_dcm[0]],
            [0., self.ref_omega, 0., ref_dcm[1]],
            [0., 0., self.ref_omega, ref_dcm[2]],
            [0., 0., 0., 2. * self.ref_omega]])
        B = -array([
            [self.ref_lambda, 0., self.ref_cop[0]],
            [0., self.ref_lambda, self.ref_cop[1]],
            [0., 0., self.ref_cop[2]],
            [0., 0., 1.]])
        self.A = A
        self.B = B
        self.K = None  # call set_gains or set_poles
        self.ref_dcm = ref_dcm
        #
        self.set_critical_gains(k_z)

    def set_poles(self, poles):
        """
        Place poles using SciPy's implementation of Kautsky et al.'s algorithm.

        Parameters
        ----------
        poles : (4,) array
            Desired poles of the closed-loop system.
        """
        bunch = scipy.signal.place_poles(self.A, self.B, poles)
        self.K = -bunch.gain_matrix  # place_poles assumes A - B * K

    def set_gains(self, gains):
        """
        Set gains from 4D DCM error to 3D input ``[zmp_x, zmp_y, lambda]``.

        Parameters
        ----------
        gains : (4,) array
            List of gains ``[k_x, k_y, k_z, k_omega]``.
        """
        k_x, k_y, k_z, k_omega = gains
        self.K = array([
            [k_x, 0., 0., 0.],
            [0., k_y, 0., 0.],
            [0., 0., k_z, k_omega]])

    def set_critical_gains(self, k_z):
        """
        Set critical gain ``k_omega`` for a desired vertical DCM gain ``k_z``.

        Parameters
        ----------
        k_z : scalar
            Desired vertical DCM to normalized leg stiffness gain.
        """
        assert k_z > 1e-10, "Feedback gain needs to be positive"
        omega = self.ref_omega
        k_xy = k_p / omega
        gamma = omega * k_p
        k_omega = omega + (k_z * self.ref_dcm[2] + gamma ** 2) / gamma
        self.set_gains([k_xy, k_xy, k_z, k_omega])

    def compute_compensation(self):
        """
        Compute CoP and normalized leg stiffness compensation.
        """
        omega = self.omega
        com = self.pendulum.com.p
        comd = self.pendulum.com.pd
        dcm = comd + omega * com
        Delta_omega = omega - self.ref_omega
        Delta_x = array([
            dcm[0] - self.ref_dcm[0],
            dcm[1] - self.ref_dcm[1],
            dcm[2] - self.ref_dcm[2],
            Delta_omega])
        Delta_u = dot(self.K, Delta_x)
        Delta_lambda = Delta_u[2]
        Delta_r = Delta_u[:2]   # contact is horizontal for now
        omegad = 2 * self.ref_omega * Delta_omega - Delta_lambda
        self.omega += omegad * sim.dt
        self.dcm = com + comd / omega
        return (Delta_r, Delta_lambda)


class Pusher(pymanoid.Process):

    """
    Send impulses to the inverted pendulum every once in a while.

    Parameters
    ----------
    pendulums : list of pymanoid.models.InvertedPendulum
        Inverted pendulums to de-stabilize.
    gain : scalar
        Magnitude of velocity jumps.

    Notes
    -----
    You know, I've seen a lot of people walkin' 'round // With tombstones in
    their eyes // But the pusher don't care // Ah, if you live or if you die
    """

    def __init__(self, pendulums, gain=0.1):
        super(Pusher, self).__init__()
        self.gain = gain
        self.handle = None
        self.mask = array([1., 1., 1.])
        self.nb_ticks = 0
        self.pendulums = pendulums
        self.started = False

    def start(self):
        self.started = True

    def stop(self):
        self.started = False

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
        one_sec = int(1. / sim.dt)
        if self.started and self.nb_ticks % one_sec == 0:
            self.push()

    def push(self, gain=None, dv=None, mask=None):
        from pymanoid.gui import draw_arrow
        if gain is None:
            gain = self.gain
        if dv is None:
            dv = 2. * numpy.random.random(3) - 1.
            if self.mask is not None:
                dv *= self.mask
            dv *= gain / numpy.linalg.norm(dv)
            print("Pusher: dv = {}".format(repr(dv)))
        arrows = []
        for pendulum in self.pendulums:
            com = pendulum.com.p
            comd = pendulum.com.pd
            pendulum.com.set_vel(comd + dv)
            arrow = draw_arrow(com - dv, com, color='b', linewidth=0.01)
            arrows.append(arrow)
        self.handle = arrows


class Plotter(pymanoid.Process):

    def __init__(self, stabilizers):
        super(Plotter, self).__init__()
        self.plots = {
            'omega': [[] for stab in stabilizers],
            'xi_x': [[] for stab in stabilizers],
            'xi_y': [[] for stab in stabilizers],
            'xi_z': [[] for stab in stabilizers]}
        self.stabilizers = stabilizers

    def on_tick(self, sim):
        for i, stab in enumerate(self.stabilizers):
            cop = stab.pendulum.cop
            dcm = stab.dcm
            omega2 = stab.omega ** 2
            lambda_ = stab.pendulum.lambda_
            self.plots['xi_x'][i].append([dcm[0], cop[0]])
            self.plots['xi_y'][i].append([dcm[1], cop[1]])
            self.plots['xi_z'][i].append([dcm[2]])
            self.plots['omega'][i].append([omega2, lambda_])

    def plot(self, size=1000):
        from pylab import clf, grid, legend, matplotlib, plot, subplot, ylim
        matplotlib.rcParams['font.size'] = 14
        legends = {
            'omega': ("$\\omega^2$", "$\\lambda$"),
            'xi_x': ("$\\xi_x$", "$z_x$"),
            'xi_y': ("$\\xi_y$", "$z_y$"),
            'xi_z': ("$\\xi_z$",)}
        clf()
        linestyles = ['-', ':', '--']
        colors = ['b', 'g', 'r']
        ref_omega = vrp_stabilizer.ref_omega
        ref_lambda = vrp_stabilizer.ref_lambda
        ref_dcm_p = vrp_stabilizer.ref_dcm
        refs = {
            'omega': [ref_omega ** 2, ref_lambda],
            'xi_x': [ref_dcm_p[0]],
            'xi_y': [ref_dcm_p[1]],
            'xi_z': [ref_dcm_p[2]]}
        for figid, figname in enumerate(self.plots):
            subplot(411 + figid)
            for i, stab in enumerate(self.stabilizers):
                curves = zip(*self.plots[figname][i][-size:])
                trange = [sim.dt * k for k in range(len(curves[0]))]
                for j, curve in enumerate(curves):
                    plot(trange, curve, linestyle=linestyles[i],
                         color=colors[j])
            for ref in refs[figname]:
                plot([trange[0], trange[-1]], [ref, ref], 'k--')
            if figname == "xi_x":
                r_x_max = contact.p[0] + contact.shape[0]
                r_x_min = contact.p[0] - contact.shape[0]
                plot([trange[0], trange[-1]], [r_x_max] * 2, 'm:', lw=2)
                plot([trange[0], trange[-1]], [r_x_min] * 2, 'm:', lw=2)
                ylim(r_x_min - 0.02, r_x_max + 0.02)
            if figname == "xi_y":
                r_y_max = contact.p[1] + contact.shape[1]
                r_y_min = contact.p[1] - contact.shape[1]
                plot([trange[0], trange[-1]], [r_y_max] * 2, 'm:', lw=2)
                plot([trange[0], trange[-1]], [r_y_min] * 2, 'm:', lw=2)
                ylim(r_y_min - 0.01, r_y_max + 0.01)
            legend(legends[figname])
            grid(True)


def push_three_times():
    """
    Apply three pushes of increasing magnitude to the CoM.

    Note
    ----
    This is the function used to generate Fig. 1 in the manuscript
    <https://hal.archives-ouvertes.fr/hal-02289919v1/document>.
    """
    sim.step(10)
    pusher.push(dv=0.4 * array([-0.04577333,  0.07776766, -0.04309285]))
    sim.step(40)
    pusher.push(dv=1.2 * array([-0.04577333,  0.07776766, -0.04309285]))
    sim.step(50)
    pusher.push(dv=1.375 * array([-0.04577333,  0.07776766, -0.04309285]))
    sim.step(100)
    plotter.plot()


class DCMPlotter(pymanoid.Process):

    def __init__(self, stabilizers):
        super(DCMPlotter, self).__init__()
        self.handles = []
        self.stabilizers = stabilizers

    def on_tick(self, sim):
        from pymanoid.gui import draw_point
        self.handles = [
            draw_point(stab.dcm, color=stab.pendulum.color, pointsize=0.01)
            for stab in self.stabilizers
            if stab.pendulum.is_visible]


def record_video():
    """
    Record accompanying video of the paper.
    """
    from pymanoid.sim import CameraRecorder
    global k_p

    k_p = 2.
    sim.set_camera_front(x=1.6, y=0, z=0.5)
    contact.hide()
    sim.contact_handle = pymanoid.gui.draw_polygon(
        [array([v[0], v[1], 0]) for v in contact.vertices],
        normal=[0, 0, 1])
    sim.max_dcm_line = pymanoid.gui.draw_line([0, 2, 1], [0, -2, 1], color='k')
    # sim.ref_line = pymanoid.gui.draw_line([0, 2, 0.8], [0, -2, 0.8])

    reading_time = 3  # [s]

    recorder = CameraRecorder(sim, "vrp_only.mp4")
    dcm_plotter = DCMPlotter(stabilizers)
    sim.schedule_extra(recorder)
    sim.schedule_extra(dcm_plotter)

    recorder.wait_for(2 * reading_time)

    vhip_stabilizer.pendulum.hide()

    dv = array([0., -0.08, 0.])
    pusher.push(dv=dv)
    print("Impulse: {} N.s".format(mass * numpy.linalg.norm(dv)))
    recorder.wait_for(reading_time)
    sim.step(1)
    recorder.wait_for(reading_time)
    sim.step(49)

    dv = array([0., -0.12, 0.])
    pusher.push(dv=dv)
    print("Impulse: {} N.s".format(mass * numpy.linalg.norm(dv)))
    recorder.wait_for(reading_time)
    sim.step(1)
    recorder.wait_for(reading_time)
    sim.step(99)

    vhip_stabilizer.pendulum.show()

    dv = array([0., -0.18, 0.])
    pusher.push(dv=dv)
    print("Impulse: {} N.s".format(mass * numpy.linalg.norm(dv)))
    recorder.wait_for(reading_time)
    sim.step(1)
    recorder.wait_for(2 * reading_time)
    sim.step(10)
    recorder.wait_for(reading_time)
    sim.step(49)
    vrp_stabilizer.pendulum.hide()
    sim.step(100)


if __name__ == '__main__':
    sim = pymanoid.Simulation(dt=0.03)
    sim.set_viewer()
    sim.viewer.SetCamera([
        [-0.28985337, 0.40434395, -0.86746239, 1.40434551],
        [0.95680245, 0.1009506, -0.27265003, 0.45636871],
        [-0.02267354, -0.90901867, -0.41613816, 1.15192068],
        [0., 0., 0., 1.]])

    contact = pymanoid.Contact((0.1, 0.05), pos=[0., 0., 0.])
    init_pos = numpy.array([0., 0., 0.8])
    init_vel = numpy.zeros(3)
    pendulums = []
    stabilizers = []

    pendulums.append(pymanoid.models.InvertedPendulum(
        init_pos, init_vel, contact, color='b', size=0.019))
    vhip_stabilizer = VHIPQPStabilizer(pendulums[-1])
    stabilizers.append(vhip_stabilizer)

    pendulums.append(pymanoid.models.InvertedPendulum(
        init_pos, init_vel, contact, color='g', size=0.02))
    # pendulums[-1].com.set_transparency(0.4)
    vrp_stabilizer = VRPStabilizer(pendulums[-1])
    stabilizers.append(vrp_stabilizer)
    # vrp_stabilizer.pendulum.hide()

    if '--bonus' in sys.argv:
        pendulums.append(pymanoid.models.InvertedPendulum(
            init_pos, init_vel, contact, color='r', size=0.015))
        bonus_stabilizer = BonusPolePlacementStabilizer(pendulums[-1], k_z=100)
        stabilizers.append(bonus_stabilizer)

    pusher = Pusher(pendulums)
    plotter = Plotter(stabilizers)

    for (stabilizer, pendulum) in zip(stabilizers, pendulums):
        sim.schedule(stabilizer)  # before pendulum
        sim.schedule(pendulum)
    sim.schedule_extra(plotter)  # before pusher
    sim.schedule_extra(pusher)

    def reset():
        for stab in stabilizers:
            stab.reset_pendulum()

    sim.step(42)  # go to reference
    push_three_times()  # scenario for Fig. 1 of the paper
    # record_video()  # video for v1 of the paper
    if IPython.get_ipython() is None:  # give the user a prompt
        IPython.embed()
