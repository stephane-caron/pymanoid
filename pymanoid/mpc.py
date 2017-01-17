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


class PreviewControl(object):

    """
    Preview control for a system with linear dynamics.

    The dynamics of a linear system are described by:

    .. math::

        x_{k+1} = A x_k + B u_k

    where :math:`x` is assumed to be the first-order state of a configuration
    variable :math:`p`, i.e., it stacks both the position :math:`p` and its
    time-derivative :math:`\\dot{p}`. Meanwhile, the system is linearly
    constrained by:

    .. math::

        \\begin{eqnarray}
        x_0 & = & x_\\mathrm{init} \\\\
        \\forall k, \\ G_k u_k & \\leq & h_k \\\\
        \\forall k, \\ E_k p_k & \\leq & f_k
        \\end{eqnarray}

    The output control law will minimize, by decreasing priority:

    - :math:`\\|x_\\mathrm{nb\\_steps} - x_\\mathrm{goal}\\|^2` with weight
      :math:`w_x`
    - :math:`\\sum_k \\|u_k\\|^2` with weight :math:`w_u`

    Where the minimization is weighted, not prioritized.

    Parameters
    ----------
    A : array, shape=(n, n)
        State linear dynamics matrix.
    B : array, shape=(n, dim(u))
        Control linear dynamics matrix.
    G : array, shape=(m, dim(u) or nb_steps * dim(u))
        Matrix for control inequality constraints.
    h : array, shape=(m,)
        Vector for control inequality constraints.
    x_init : array, shape=(n,)
        Initial state as stacked position and velocity.
    x_goal : array, shape=(n,)
        Goal state as stacked position and velocity.
    nb_steps : int
        Number of discretization steps in the preview window.
    E : array, shape=(l, n), optional
        Matrix for state inequality constraints.
    f : array, shape=(l,), optional
        Vector for state inequality constraints.
    wx : scalar, optional, default=1000.
        Weight :math:`w_x` on the state error
        :math:`\\|x - x_\\mathrm{goal}\\|^2`.
    wu : scalar, optional, default=1.
        Weight :math:`w_u` on cumulated controls
        :math:`\\sum_k \\|u_k\\|^2`.
    """

    def __init__(self, A, B, G, h, x_init, x_goal, nb_steps, E=None, f=None,
                 wx=1000., wu=1.):
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        assert G.shape[1] in [u_dim, nb_steps * u_dim]
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
        self.t_build_start = None
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.x_goal = x_goal
        self.x_init = x_init
        self.wu = wu
        self.wx = wx

    def compute_dynamics(self):
        """
        Compute internal matrices mapping stacked controls ``U`` to states.

        Notes
        -----
        See [Aud+14]_ for details, as we use the same notations below.

        References
        ----------
        .. [Aud+14] Hervé Audren, Joris Vaillant, Aberrahmane Kheddar, Adrien
            Escande, Kenji Kaneko, Eiichi Yoshida, "Model preview control in
            multi-contact motion-application to a humanoid robot," 2014 IEEE/RSJ
            International Conference on Intelligent Robots and Systems, Chicago,
            IL, 2014, pp. 4030-4035.
            `[doi]
            <http://dx.doi.org/10.1109/IROS.2014.6943129>`__
            `[pdf]
            <https://staff.aist.go.jp/e.yoshida/papers/Audren_iros2014.pdf>`__
        """
        self.t_build_start = time()
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
        Compute the stacked control vector `U` minimizing the preview QP.
        """
        assert self.psi_last is not None, "Call compute_dynamics() first"

        # Cost 1: sum_k u_k^2
        Pu = eye(self.U_dim)
        qu = zeros(self.U_dim)
        wu = self.wu

        # Cost 2: |x_N - x_goal|^2 = |A * x - b|^2
        A = self.psi_last
        b = self.x_goal - dot(self.phi_last, self.x_init)
        Px = dot(A.T, A)
        qx = -dot(b.T, A)
        wx = self.wx

        # Weighted combination of both costs
        P = wx * Px + wu * Pu
        q = wx * qx + wu * qu

        # Inequality constraints
        G = self.G if self.E is None else vstack([self.G] + self.G_state)
        h = self.h if self.E is None else hstack([self.h] + self.h_state)
        assert self.E is None
        t_solve_start = time()
        self.U = solve_qp(P, q, G, h)
        t_done = time()
        self.solve_time = t_done - t_solve_start
        self.solve_and_build_time = t_done - self.t_build_start


try:
    from minieigen import VectorXd

    import mpcontroller as vsmpc

    from misc import array_to_MatrixXd, array_to_VectorXd, VectorXd_to_array

    class VSPreviewControl(PreviewControl):

        """
        Wrapper to Vincent Samy's preview controller.

        Source code and installation instructions are available from
        <https://github.com/vsamy/preview_controller>.

        Parameters
        ----------
        A : array, shape=(n, n)
            State linear dynamics matrix.
        B : array, shape=(n, dim(u))
            Control linear dynamics matrix.
        G : array, shape=(m, dim(u) or nb_steps * dim(u))
            Matrix for control inequality constraints.
        h : array, shape=(m,)
            Vector for control inequality constraints.
        x_init : array, shape=(n,)
            Initial state, i.e. stacked position and velocity.
        x_goal : array, shape=(n,)
            Goal state, i.e. stacked position and velocity.
        nb_steps : int
            Number of discretization steps in the preview window.
        E : array, shape=(l, n), optional
            Matrix for state inequality constraints.
        f : array, shape=(l,), optional
            Vector for state inequality constraints.
        wx : scalar, optional
            Weight :math:`w_x` on the state error
            :math:`\\|x - x_\\mathrm{goal}\\|^2`.
        wu : scalar, optional
            Weight :math:`w_u` on cumulated controls
            :math:`\\sum_k \\|u_k\\|^2`.
        solver : vsmpc.SolverFlag, optional
            Backend QP solver to use.
        """

        def __init__(self, A, B, G, h, x_init, x_goal, nb_steps, E=None, f=None,
                     wx=1000., wu=1., solver=vsmpc.SolverFlag.QuadProgDense):
            self.A = array_to_MatrixXd(A)
            self.B = array_to_MatrixXd(B)
            self.G = array_to_MatrixXd(G)
            self.c = VectorXd.Zero(A.shape[0])  # no bias term for now
            self.controller = None
            self.debug = False
            self.h = array_to_VectorXd(h)
            self.nb_steps = nb_steps
            self.ps = None
            self.solver = solver
            self.x_goal = array_to_VectorXd(x_goal)
            self.x_init = array_to_VectorXd(x_init)
            self.wu = VectorXd.Ones(B.shape[1]) * wu
            self.wx = VectorXd.Ones(A.shape[1]) * wx

        def compute_dynamics(self):
            """
            Compute internal matrices defining the preview QP.
            """
            self.ps = vsmpc.NewPreviewSystem()
            self.ps.system(
                self.A, self.B, self.c, self.x_init, self.x_goal, self.nb_steps)
            self.controller = vsmpc.MPCTypeLast(self.ps, self.solver)
            self.control_ineq = vsmpc.NewControlConstraint(self.G, self.h, True)
            if self.debug:
                print "A =", repr(self.A)
                print "B =", repr(self.B)
                print "c =", repr(self.c)
                print "x_init =", repr(self.x_init)
                print "x_goal =", repr(self.x_goal)
                print "nb_steps =", repr(self.nb_steps)
                print "G =", repr(self.G)
                print "h =", repr(self.h)
                print "wu =", repr(self.wu)
                print "wx =", repr(self.wx)
            self.controller.addConstraint(self.control_ineq)
            self.controller.weights(self.wx, self.wu)

        def compute_control(self):
            """
            Compute the stacked control vector ``U`` minimizing the preview QP.
            """
            assert self.controller is not None, "Call compute_dynamics() first"
            ret = self.controller.solve()
            if not ret:
                raise Exception("MPC failed to solve QP")
            self.U = VectorXd_to_array(self.controller.control())
            self.solve_time = self.controller.solveTime().wall  # in [ns]
            self.solve_and_build_time = self.controller.solveAndBuildTime().wall
            self.solve_time *= 1e-9  # in [s]
            self.solve_and_build_time *= 1e-9  # in [s]

except ImportError:  # mpcontroller module not available
    pass
