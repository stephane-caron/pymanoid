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

from numpy import dot, eye, hstack, vstack, zeros

from optim import solve_qp
from time import time as time


class LinearPredictiveControl(object):

    """
    Predictive control for a system with linear dynamics and linear constraints.

    The discretized dynamics of a linear system are described by:

    .. math::

        x_{k+1} = A x_k + B u_k

    where :math:`x` is assumed to be the first-order state of a configuration
    variable :math:`p`, i.e., it stacks both the position :math:`p` and its
    time-derivative :math:`\\dot{p}`. Meanwhile, the system is linearly
    constrained by:

    .. math::

        \\begin{eqnarray}
        x_0 & = & x_\\mathrm{init} \\\\
        \\forall k, \\ C_k x_k + D_k u_k & \\leq & e_k \\\\
        \\end{eqnarray}

    The output control law minimizes a weighted combination of two types of
    costs:

    - Terminal state error
        :math:`\\|x_\\mathrm{nb\\_steps} - x_\\mathrm{goal}\\|^2`
        with weight :math:`w_{xt}`.
    - Cumulated state error:
        :math:`\\sum_k \\|x_k - x_\\mathrm{goal}\\|^2`
        with weight :math:`w_{xc}`.
    - Cumulated control costs:
        :math:`\\sum_k \\|u_k\\|^2`
        with weight :math:`w_{u}`.

    Parameters
    ----------
    A : array, shape=(n, n)
        State linear dynamics matrix.
    B : array, shape=(n, dim(u))
        Control linear dynamics matrix.
    x_init : array, shape=(n,)
        Initial state as stacked position and velocity.
    x_goal : array, shape=(n,)
        Goal state as stacked position and velocity.
    nb_steps : int
        Number of discretization steps in the preview window.
    C : array, shape=(m, dim(u)), list of arrays, or None
        Constraint matrix on state variables. When this argument is an array,
        the same matrix `C` is applied at each step `k`. When it is ``None``,
        the null matrix is applied.
    D : array, shape=(l, n), or list of arrays, or None
        Constraint matrix on control variables. When this argument is an array,
        the same matrix `D` is applied at each step `k`. When it is ``None``,
        the null matrix is applied.
    e : array, shape=(m,), list of arrays
        Constraint vector. When this argument is an array, the same vector `e`
        is applied at each step `k`.
    wxt : scalar, optional
        Weight on terminal state cost, or ``None`` to disable.
    wxc : scalar, optional
        Weight on cumulated state costs, or ``None`` to disable (default).
    wu : scalar, optional
        Weight on cumulated control costs.

    Notes
    -----
    In numerical analysis, there are three classes of methods to solve `boundary
    value problems <https://en.wikipedia.org/wiki/Boundary_value_problem>`_:
    single shooting, multiple shooting and collocation. The solver implemented
    in this class follows the `single shooting method
    <https://en.wikipedia.org/wiki/Shooting_method>`_.
    """

    def __init__(self, A, B, C, D, e, x_init, x_goal, nb_steps, wxt=None,
                 wxc=None, wu=1e-3):
        assert C is not None or D is not None, "use LQR for unconstrained case"
        assert wu > 0., "non-negative control weight needed for regularization"
        assert wxt is not None or wxc is not None, "set either wxt or wxc"
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.G = None
        self.P = None
        self.U = None
        self.U_dim = u_dim * nb_steps
        self.__X = None
        self.build_time = None
        self.e = e
        self.h = None
        self.nb_steps = nb_steps
        self.q = None
        self.solve_time = None
        self.u_dim = u_dim
        self.wu = wu
        self.wxc = wxc
        self.wxt = wxt
        self.x_dim = x_dim
        self.x_goal = x_goal
        self.x_init = x_init

    @property
    def solve_and_build_time(self):
        return self.build_time + self.solve_time

    def build(self):
        """
        Compute internal matrices defining the preview QP.

        Notes
        -----
        See [Audren14]_ for details on the matrices :math:`\\Phi` and
        :math:`\\Psi`, as we use similar notations below.
        """
        t_build_start = time()
        phi = eye(self.x_dim)
        psi = zeros((self.x_dim, self.U_dim))
        G_list, h_list = [], []
        phi_list, psi_list = [], []
        for k in xrange(self.nb_steps):
            # Loop invariant: x == psi * U + phi * x_init
            if self.wxc is not None:
                phi_list.append(phi)
                psi_list.append(psi)
            C = self.C[k] if type(self.C) is list else self.C
            D = self.D[k] if type(self.D) is list else self.D
            e = self.e[k] if type(self.e) is list else self.e
            G = zeros((e.shape[0], self.U_dim))
            h = e if C is None else e - dot(dot(C, phi), self.x_init)
            if D is not None:
                # we rely on G == 0 to avoid a slower +=
                G[:, k * self.u_dim:(k + 1) * self.u_dim] = D
            if C is not None:
                G += dot(C, psi)
            G_list.append(G)
            h_list.append(h)
            phi = dot(self.A, phi)
            psi = dot(self.A, psi)
            psi[:, self.u_dim * k:self.u_dim * (k + 1)] = self.B
        P = self.wu * eye(self.U_dim)
        q = zeros(self.U_dim)
        if self.wxt is not None and self.wxt > 1e-10:
            c = dot(phi, self.x_init) - self.x_goal
            P += self.wxt * dot(psi.T, psi)
            q += self.wxt * dot(c.T, psi)
        if self.wxc is not None and self.wxc > 1e-10:
            Phi = vstack(phi_list)
            Psi = vstack(psi_list)
            X_goal = hstack([self.x_goal] * self.nb_steps)
            c = dot(Phi, self.x_init) - X_goal
            P += self.wxc * dot(Psi.T, Psi)
            q += self.wxc * dot(c.T, Psi)
        self.P = P
        self.q = q
        self.G = vstack(G_list)
        self.h = hstack(h_list)
        self.build_time = time() - t_build_start

    def solve(self):
        """
        Compute the series of controls that minimizes the preview QP.

        Note
        ----
        This function can only be called after ``build()``.
        """
        assert self.P is not None, "you need to build() the MPC problem first"
        t_solve_start = time()
        U = solve_qp(self.P, self.q, self.G, self.h)
        self.U = U.reshape((self.nb_steps, self.u_dim))
        self.solve_time = time() - t_solve_start

    @property
    def X(self):
        """
        Series of system states over the preview window.

        Note
        ----
        This property is only available after ``solve()`` has been called.
        """
        if self.__X is not None:
            return self.__X
        assert self.U is not None, "you need to solve() the MPC problem first"
        X = zeros((self.nb_steps + 1, self.x_dim))
        X[0] = self.x_init
        for k in xrange(self.nb_steps):
            X[k + 1] = dot(self.A, X[k]) + dot(self.B, self.U[k])
        self.__X = X
        return X
