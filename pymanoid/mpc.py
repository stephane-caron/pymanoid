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

from numpy import dot, eye, hstack, ndarray, vstack, zeros

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
        \\forall k, \\ C_k u_k & \\leq & d_k \\\\
        \\forall k, \\ E_k p_k & \\leq & f_k
        \\end{eqnarray}

    The output control law minimizes a weighted combination of two types of
    costs:

    - Terminal state error
        :math:`\\|x_\\mathrm{nb\\_steps} - x_\\mathrm{goal}\\|^2`
      or cumulated state error:
        :math:`\\sum_k \\|x_k - x_\\mathrm{goal}\\|^2`
    - Cumulated control costs:
        :math:`\\sum_k \\|u_k\\|^2`

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
    C : array, shape=(m, dim(u)), or list of arrays, optional
        Matrices for control inequality constraints. When this argument is an
        array, the same matrix `C` is applied at each step `k`.
    d : array, shape=(m,), or list of arrays, optional
        Vectors for control inequality constraints. When this argument is an
        array, the same vector `d` is applied at each step `k`.
    E : array, shape=(l, n), or list of arrays, optional
        Matrix for state inequality constraints. When this argument is an
        array, the same matrix `E` is applied at each step `k`.
    f : array, shape=(l,), or list of arrays, optional
        Vector for state inequality constraints. When this argument is an array,
        the same vector `f` is applied at each step `k`.
    state_cost : string, optional
        Switch between "terminal" or "cumulated" state costs.

    Notes
    -----
    In numerical analysis, there are three classes of methods to solve `boundary
    value problems <https://en.wikipedia.org/wiki/Boundary_value_problem>`_:
    single shooting, multiple shooting and collocation. The solver implemented
    in this class follows the `single shooting method
    <https://en.wikipedia.org/wiki/Shooting_method>`_.
    """

    def __init__(self, A, B, x_init, x_goal, nb_steps, C=None, d=None, E=None,
                 f=None, state_cost='terminal'):
        assert state_cost in ['cumulated', 'terminal']
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.Phi = None
        self.Psi = None
        self.U = None
        self.U_dim = u_dim * nb_steps
        self.d = d
        self.f = f
        self.is_terminal = state_cost == 'terminal'
        self.nb_steps = nb_steps
        self.phi_last = None
        self.psi_last = None
        self.t_build_start = None
        self.u_dim = u_dim
        self.x_dim = x_dim
        self.x_goal = x_goal
        self.x_init = x_init

    def compute_dynamics(self):
        """
        Compute internal matrices defining the preview QP.

        Notes
        -----
        See [Aud+14]_ for details on the matrices :math:`\\Phi` and
        :math:`\\Psi`, as we use similar notations below.

        References
        ----------
        .. [Aud+14] Hervé Audren, Joris Vaillant, Aberrahmane Kheddar, Adrien
            Escande, Kenji Kaneko, Eiichi Yoshida, "Model preview control in
            multi-contact motion-application to a humanoid robot," 2014 IEEE/RSJ
            International Conference on Intelligent Robots and Systems, Chicago,
            IL, 2014, pp. 4030-4035.
            `[doi]
            <http://doai.io/10.1109/IROS.2014.6943129>`__
            `[pdf]
            <https://staff.aist.go.jp/e.yoshida/papers/Audren_iros2014.pdf>`__
        """
        self.t_build_start = time()
        phi = eye(self.x_dim)
        psi = zeros((self.x_dim, self.U_dim))
        G_list, h_list = [], []
        phi_list, psi_list = [], []
        for k in xrange(self.nb_steps):
            # Loop invariant: x = phi * x_init + psi * U
            if not self.is_terminal:
                phi_list.append(phi)
                psi_list.append(psi)
            if self.C is not None:
                # {C * u <= d} iff {C_ext * U <= d}
                C = self.C if type(self.C) is ndarray else self.C[k]
                d = self.d if type(self.d) is ndarray else self.d[k]
                C_ext = zeros((C.shape[0], self.U_dim))
                C_ext[:, k * self.u_dim:(k + 1) * self.u_dim] = C
                G_list.append(C_ext)
                h_list.append(d)
            if self.E is not None:
                # {E * x <= f} iff {(E * psi) * U <= f - (E * phi) * x_init}
                E = self.E if type(self.E) is ndarray else self.E[k]
                f = self.f if type(self.f) is ndarray else self.f[k]
                G_list.append(dot(E, psi))
                h_list.append(f - dot(dot(E, phi), self.x_init))
            phi = dot(self.A, phi)
            psi = dot(self.A, psi)
            psi[:, self.u_dim * k:self.u_dim * (k + 1)] = self.B
        if self.is_terminal:
            self.phi_last = phi
            self.psi_last = psi
        else:  # not self.is_terminal:
            self.Phi = vstack(phi_list)
            self.Psi = vstack(psi_list)
        self.G = vstack(G_list)
        self.h = hstack(h_list)

    def compute_controls(self, wx=1., wu=1e-3):
        """
        Compute the series of controls that minimizes the preview QP.

        Parameters
        ----------
        wx : scalar, optional
            Weight on (cumulated or terminal) state costs.
        wu : scalar, optional
            Weight on cumulated control costs.

        Note
        ----
        This function should be called after ``compute_dynamics()``.
        """
        P = wu * eye(self.U_dim)
        q = zeros(self.U_dim)
        if self.is_terminal:
            A = self.psi_last
            b = dot(self.phi_last, self.x_init) - self.x_goal
        else:  # not self.is_terminal:
            A = self.Psi
            b = dot(self.Phi, self.x_init)
        P += wx * dot(A.T, A)
        q += wx * dot(b.T, A)
        t_solve_start = time()
        U = solve_qp(P, q, self.G, self.h)
        t_done = time()
        self.U = U.reshape((self.nb_steps, self.u_dim))
        self.solve_time = t_done - t_solve_start
        self.solve_and_build_time = t_done - self.t_build_start

    def compute_states(self):
        """
        Compute the series of system states over the preview window.

        Note
        ----
        This function should be called after ``compute_controls()``.
        """
        assert self.U is not None, "call compute_controls() first"
        X = zeros((self.nb_steps + 1, self.x_dim))
        X[0] = self.x_init
        for k in xrange(self.nb_steps):
            X[k + 1] = dot(self.A, X[k]) + dot(self.B, self.U[k])
        self.X = X


try:
    from minieigen import VectorXd

    import mpcontroller as vsmpc

    from misc import array_to_MatrixXd, array_to_VectorXd, VectorXd_to_array

    class VSLMPC(LinearPredictiveControl):

        """
        Wrapper to Vincent Samy's LMPC library.

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
            u_dim = B.shape[1]
            self.A = array_to_MatrixXd(A)
            self.B = array_to_MatrixXd(B)
            self.G = array_to_MatrixXd(G)
            self.c = VectorXd.Zero(A.shape[0])  # no bias term for now
            self.controller = None
            self.h = array_to_VectorXd(h)
            self.nb_steps = nb_steps
            self.ps = None
            self.solver = solver
            self.u_dim = u_dim
            self.wu = VectorXd.Ones(B.shape[1]) * wu
            self.wx = VectorXd.Ones(A.shape[1]) * wx
            self.x_goal = array_to_VectorXd(x_goal)
            self.x_init = array_to_VectorXd(x_init)

        def compute_dynamics(self):
            """
            Compute internal matrices defining the preview QP.
            """
            self.ps = vsmpc.NewPreviewSystem()
            self.ps.system(
                self.A, self.B, self.c, self.x_init, self.x_goal, self.nb_steps)
            self.controller = vsmpc.MPCTypeLast(self.ps, self.solver)
            self.control_ineq = vsmpc.NewControlConstraint(self.G, self.h, True)
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
            U = VectorXd_to_array(self.controller.control())
            self.U = U.reshape((self.nb_steps, self.u_dim))
            self.solve_time = self.controller.solveTime().wall  # in [ns]
            self.solve_and_build_time = self.controller.solveAndBuildTime().wall
            self.solve_time *= 1e-9  # in [s]
            self.solve_and_build_time *= 1e-9  # in [s]

except ImportError:  # mpcontroller module not available
    pass
