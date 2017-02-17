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

    def __init__(self, A, B, x_init, x_goal, nb_steps, C=None, d=None, E=None,
                 f=None, wxt=None, wxc=None, wu=1e-3):
        assert C is not None or E is not None, "use LQR for unconstrained case"
        assert wu > 0., "non-negative control weight needed for regularization"
        assert wxt is not None or wxc is not None, "set either wxt or wxc"
        u_dim = B.shape[1]
        x_dim = A.shape[1]
        self.A = A
        self.B = B
        self.C = C
        self.E = E
        self.G = None
        self.U = None
        self.U_dim = u_dim * nb_steps
        self.build_time = None
        self.d = d
        self.f = f
        self.h = None
        self.nb_steps = nb_steps
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
        t_build_start = time()
        phi = eye(self.x_dim)
        psi = zeros((self.x_dim, self.U_dim))
        G_list, h_list = [], []
        phi_list, psi_list = [], []
        for k in xrange(self.nb_steps):
            # Loop invariant: x = phi * x_init + psi * U
            if self.wxc is not None:
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
        P = self.wu * eye(self.U_dim)
        q = zeros(self.U_dim)
        if self.wxt is not None:
            c = dot(phi, self.x_init) - self.x_goal
            P += self.wxt * dot(psi.T, psi)
            q += self.wxt * dot(c.T, psi)
        if self.wxc is not None:
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
        """
        t_solve_start = time()
        U = solve_qp(self.P, self.q, self.G, self.h)
        self.U = U.reshape((self.nb_steps, self.u_dim))
        self.solve_time = time() - t_solve_start

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
        x_init : array, shape=(n,)
            Initial state, i.e. stacked position and velocity.
        x_goal : array, shape=(n,)
            Goal state, i.e. stacked position and velocity.
        nb_steps : int
            Number of discretization steps in the preview window.
        C : array, shape=(m, dim(u) or nb_steps * dim(u))
            Matrix for control inequality constraints.
        d : array, shape=(m,)
            Vector for control inequality constraints.
        E : array, shape=(l, n), optional
            Matrix for state inequality constraints.
        f : array, shape=(l,), optional
            Vector for state inequality constraints.
        solver : vsmpc.SolverFlag, optional
            Backend QP solver to use.
        """

        def __init__(self, A, B, x_init, x_goal, nb_steps, C=None, d=None,
                     E=None, f=None, solver=vsmpc.SolverFlag.QuadProgDense):
            self.A = array_to_MatrixXd(A)
            self.B = array_to_MatrixXd(B)
            self.C = array_to_MatrixXd(C)
            self.c = VectorXd.Zero(A.shape[0])  # no bias term for now
            self.controller = None
            self.d = array_to_VectorXd(d)
            self.nb_steps = nb_steps
            self.ps = None
            self.solver = solver
            self.u_dim = B.shape[1]
            self.x_dim = A.shape[1]
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
            self.control_ineq = vsmpc.NewControlConstraint(self.C, self.d, True)
            self.controller.addConstraint(self.control_ineq)

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
            assert self.controller is not None, "Call compute_dynamics() first"
            wu = VectorXd.Ones(self.u_dim) * wu
            wx = VectorXd.Ones(self.x_dim) * wx
            self.controller.weights(wx, wu)
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
