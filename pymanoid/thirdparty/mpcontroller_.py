#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with pymanoid. If not, see <http://www.gnu.org/licenses/>.

from minieigen import MatrixXd, VectorXd
from numpy import array

from mpcontroller import MPCTypeLast, NewControlConstraint, NewPreviewSystem
from mpcontroller import SolverFlag


def array_to_MatrixXd(a):
    A = MatrixXd.Zero(a.shape[0], a.shape[1])
    for i in xrange(a.shape[0]):
        for j in xrange(a.shape[1]):
            A[i, j] = a[i, j]
    return A


def array_to_VectorXd(v):
    V = VectorXd.Zero(v.shape[0])
    for i in xrange(v.shape[0]):
        V[i] = v[i]
    return V


def VectorXd_to_array(V):
    return array([V[i] for i in xrange(V.rows())])


class LinearPredictiveControl(object):

    """
    Wrapper to Vincent Samy's LMPC library. The API of this class is the same as
    the vanilla :class:`pymanoid.mpc.LinearPredictiveControl`. Source code and
    installation instructions for the library are available from
    <https://github.com/vsamy/preview_controller>.

    The discretized dynamics of a linear system are written as:

    .. math::

        x_{k+1} = A x_k + B u_k

    where :math:`x` is assumed to be the first-order state of a
    configuration variable :math:`p`, i.e., it stacks both the position
    :math:`p` and its time-derivative :math:`\\dot{p}`. Meanwhile, the
    system is linearly constrained by:

    .. math::

        \\begin{eqnarray}
        x_0 & = & x_\\mathrm{init} \\\\
        \\forall k, \\ C_k u_k & \\leq & d_k \\\\
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
        Initial state, i.e. stacked position and velocity.
    x_goal : array, shape=(n,)
        Goal state, i.e. stacked position and velocity.
    nb_steps : int
        Number of discretization steps in the preview window.
    C : array, shape=(m, dim(u) or nb_steps * dim(u))
        Matrix for control inequality constraints.
    d : array, shape=(m,)
        Vector for control inequality constraints.
    solver : SolverFlag, optional
        Backend QP solver to use.
    """

    def __init__(self, A, B, x_init, x_goal, nb_steps, C=None, d=None,
                 solver=SolverFlag.QuadProgDense):
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
        self.ps = NewPreviewSystem()
        self.ps.system(
            self.A, self.B, self.c, self.x_init, self.x_goal, self.nb_steps)
        self.controller = MPCTypeLast(self.ps, self.solver)
        self.control_ineq = NewControlConstraint(self.C, self.d, True)
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
