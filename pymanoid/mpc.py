#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Stephane Caron <stephane.caron@normalesup.org>
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

from numpy import dot, eye, hstack, vstack, zeros
from threading import Lock

from process import Process
from optim import solve_qp


class PreviewBuffer(Process):

    """
    Buffer to store controls output by the preview controller.
    """

    def __init__(self, callback):
        """
        Create a new buffer associated with a given target.

        INPUT:

        - ``callback`` -- function to call with each new control ``(u, dT)``
        """
        super(PreviewBuffer, self).__init__()
        self.callback = callback
        self.cur_control = None
        self.preview = None
        self.preview_index = 0
        self.preview_lock = Lock()
        self.rem_time = 0.

    def update_preview(self, preview):
        """
        Update preview with a filled PreviewControl object.
        """
        with self.preview_lock:
            self.preview_index = 0
            self.preview = preview

    def get_next_control(self):
        """
        Return the next pair ``(u, dT)`` in the preview window.
        """
        with self.preview_lock:
            if self.preview is None:
                return (zeros(3), 0.)
            j = 3 * self.preview_index
            u = self.preview.U[j:j + 3]
            if u.shape[0] == 0:
                self.preview = None
                return (zeros(3), 0.)
            dT = self.preview.timestep
            self.preview_index += 1
            return (u, dT)

    def on_tick(self, sim):
        """
        Entry point called at each simulation tick.
        """
        if self.rem_time < sim.dt:
            u, dT = self.get_next_control()
            self.cur_control = u
            self.rem_time = dT
        self.callback(self.cur_control, sim.dt)
        self.rem_time -= sim.dt


class PreviewControl(object):

    """
    Preview control for a system with linear dynamics:

        x_{k+1} = A * x_k + B * u_k

    where x is assumed to be the state of a configuration variable p, i.e.,

        x_k = [  p_k ]
              [ pd_k ]

    subject to constraints:

        x_0 = x_init                    -- initial state
        for all k,   C(k) * u_k <= d(k) -- control constraints
        for all k,   E(k) * p_k <= f(k) -- position constraints

    The output control law will minimize, by decreasing priority:

        1)  |x_{nb_steps} - x_goal|^2
        2)  sum_k |u_k|^2

    The minimization is weighted (not prioritized).
    """

    def __init__(self, A, B, G, h, x_init, x_goal, nb_steps, E=None, f=None):
        """
        Create a new preview controller.

        INPUT:

        - ``A`` -- state linear dynamics matrix
        - ``B`` -- control linear dynamics matrix
        - ``G`` -- matrix for control inequality constraints
        - ``h`` -- vector for control inequality constraints
        - ``x_init`` -- initial state
        - ``x_goal`` -- goal state
        - ``nb_steps`` -- number of discretized time steps
        - ``E`` -- (optional) matrix for state inequality constraints
        - ``f`` -- (optional) vector for state inequality constraints
        """
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.E = E
        self.G = G
        self.U_dim = u_dim * nb_steps
        self.X_dim = x_dim * nb_steps  # not used but meh
        self.f = f
        self.h = h
        self.nb_steps = nb_steps
        self.phi_last = None
        self.psi_last = None
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.x_goal = x_goal
        self.x_init = x_init

    def compute_dynamics(self):
        """
        Compute internal matrices mapping stacked controls ``U`` to states.

        ALGORITHM:

        See [Aud+14] for details, as we use the same notations below.

        REFERENCES:

        .. [Aud+14] http://dx.doi.org/10.1109/IROS.2014.6943129
        """
        phi = eye(self.x_dim)
        psi = zeros((self.x_dim, self.U_dim))
        G_list, h_list = [], []
        for k in xrange(self.nb_steps):
            # x_k = phi * x_init + psi * U
            # p_k = phi[:3] * x_init + psi[:3] * U
            # E * p_k <= f
            # (E * psi[:3]) * U <= f - (E * phi[:3]) * x_init
            if self.E is not None:
                G_list.append(dot(self.E, psi[:3]))
                h_list.append(self.f - dot(dot(self.E, phi[:3]), self.x_init))
            phi = dot(self.A, phi)
            psi = dot(self.A, psi)
            psi[:, self.u_dim * k:self.u_dim * (k + 1)] = self.B
        self.G_state = G_list
        self.h_state = h_list
        self.phi_last = phi
        self.psi_last = psi

    def compute_control(self):
        """
        Compute the stacked control vector ``U`` minimizing the preview QP.
        """
        assert self.psi_last is not None, "Call compute_dynamics() first"

        # Cost 1: sum_k u_k^2
        P1 = eye(self.U_dim)
        q1 = zeros(self.U_dim)
        w1 = 1.

        # Cost 2: |x_N - x_goal|^2 = |A * x - b|^2
        A = self.psi_last
        b = self.x_goal - dot(self.phi_last, self.x_init)
        P2 = dot(A.T, A)
        q2 = -dot(b.T, A)
        w2 = 1000.

        # Weighted combination of both costs
        P = w1 * P1 + w2 * P2
        q = w1 * q1 + w2 * q2

        # Inequality constraints
        G = self.G if self.E is None else vstack([self.G] + self.G_state)
        h = self.h if self.E is None else hstack([self.h] + self.h_state)
        self.U = solve_qp(P, q, G, h)
