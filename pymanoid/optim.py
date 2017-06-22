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

from numpy import eye, hstack, ones, vstack, zeros

from thirdparty.cvxopt_ import solve_lp
from thirdparty.cvxopt_ import solve_qp as cvxopt_solve_qp
from thirdparty.quadprog_ import solve_qp as quadprog_solve_qp

try:
    from thirdparty.mosek_ import solve_qp as mosek_solve_qp
    MOSEK_IF_AVAILABLE = 'mosek'
except ImportError:
    MOSEK_IF_AVAILABLE = None


def solve_qp(P, q, G, h, A=None, b=None, solver='quadprog'):
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} & & G x \leq h \\\\
            & & A x = b
        \\end{eqnarray}

    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    solver : string, optional
        Name of the QP solver to use (default is quadprog).

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the QP, if found.

    Raises
    ------
    ValueError
        If the QP is not feasible.
    """
    if solver == 'cvxopt':
        return cvxopt_solve_qp(P, q, G, h, A, b)
    elif solver == 'mosek':
        return mosek_solve_qp(P, q, G, h, A, b)
    elif solver == 'quadprog':
        return quadprog_solve_qp(P, q, G, h, A, b)
    raise Exception("QP solver '%s' not recognized" % solver)


def solve_safer_qp(P, q, G, h, w_reg, w_lin, solver=MOSEK_IF_AVAILABLE):
    """
    Solve the relaxed Quadratic Program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x
        + (1/2) \\epsilon \\| s \\|^2 - w_\\mathrm{lin} 1^T s\\\\
        \\mathrm{subject\\ to} & & G x + s \\leq h \\\\
            & & s \\geq 0
        \\end{eqnarray}

    Slack variables `s` are increased by an additional term in the cost
    function, so that the solution of this "safer" QP is further inside the
    constraint region.

    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    w_reg : scalar
        Regularization term :math:`(1/2) \\epsilon` in the cost function. Set
        this parameter as small as possible (e.g. 1e-8), and increase it in case
        of numerical instability.
    w_lin : scalar
        Weight of the linear cost on slack variables. Higher values bring the
        solution further inside the constraint region but override the
        minimization of the original objective.
    solver : string, optional
        Name of the QP solver to use (default is MOSEK).

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the relaxed QP, if found.

    Raises
    ------
    ValueError
        If the QP is not feasible.

    Notes
    -----
    The default solver for this method is MOSEK, as it seems to perform better
    on such relaxed problems.
    """
    n, m = P.shape[0], G.shape[0]
    E, Z = eye(m), zeros((m, n))
    P2 = vstack([hstack([P, Z.T]), hstack([Z, w_reg * eye(m)])])
    q2 = hstack([q, -w_lin * ones(m)])
    G2 = hstack([Z, E])
    h2 = zeros(m)
    A2 = hstack([G, -E])
    b2 = h
    return solve_qp(P2, q2, G2, h2, A2, b2, solver=solver)[:n]


__all__ = ['solve_lp', 'solve_qp', 'solve_safer_qp']

try:
    from thirdparty.casadi_ import NonlinearProgram
    __all__ = ['solve_lp', 'solve_qp', 'solve_safer_qp', 'NonlinearProgram']
except ImportError:
    pass
