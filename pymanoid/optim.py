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

import cvxopt
import cvxopt.solvers

from cvxopt import matrix as cvxmat
from cvxopt.solvers import lp as cvxopt_lp
from cvxopt.solvers import qp as cvxopt_qp
from numpy import array, eye, hstack, ones, vstack, zeros

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output


"""
Linear Programming
==================
"""

try:
    import cvxopt.glpk
    LP_SOLVER = 'glpk'
    # GLPK is the fastest LP solver I could find so far:
    # <https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html>
    # ... however, it's verbose by default, so tell it to STFU:
    cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
    cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # cvxopt 1.1.7
    cvxopt.solvers.options['LPX_K_MSGLEV'] = 0  # previous versions
except ImportError:
    print "\033[1;33m[pymanoid] Warning: GLPK solver not found\033[0;0m"
    LP_SOLVER = None


def solve_lp(c, G, h, A=None, b=None, solver=LP_SOLVER):
    """
    Solve a Linear Program defined by:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & c^T x \\\\
        \\mathrm{subject\\ to} & & G x \leq h \\\\
            & & A x = b
        \\end{eqnarray}

    using CVXOPT <http://cvxopt.org/userguide/coneprog.html#linear-programming>.

    Parameters
    ----------
    c : array, shape=(n,)
        Cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    solver : string, optional
        Solver to use, default is GLPK if available

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the LP, if found, otherwise ``None``.

    Raises
    ------
    ValueError
        If the LP is not feasible.
    """
    args = [cvxmat(c), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = cvxopt_lp(*args, solver=solver)
    if 'optimal' not in sol['status']:
        raise ValueError("LP optimum not found: %s" % sol['status'])
    return array(sol['x']).reshape((c.shape[0],))


"""
Quadratic Programming
=====================
"""

try:
    # quadprog is the fastest QP solver I could find so far
    from quadprog import solve_qp as _quadprog_solve_qp

    def quadprog_solve_qp(P, q, G, h, A=None, b=None):
        """
        Solve a Quadratic Program defined as:

        .. math::

            \\begin{eqnarray}
            \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
            \\mathrm{subject\\ to} & & G x \leq h \\\\
                & & A x = b
            \\end{eqnarray}

        using the `quadprog <https://pypi.python.org/pypi/quadprog/>`_ QP
        solver, which implements the Goldfarb-Idnani dual algorithm [GI83]_.

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

        Returns
        -------
        x : array, shape=(n,)
            Optimal solution to the QP, if found.

        Raises
        ------
        ValueError
            If the QP is not feasible.

        References
        ----------
        .. [GI83] D. Goldfarb and A. Idnani, "A numerically stable dual method
            for solving strictly convex quadratic programs," Mathematical
            Programming, vol. 27, pp. 1-33, 1983.
            `[doi]
            <http://doai.io/10.1007/BF02591962>`__
            `[pdf]
            <http://download.springer.com/static/pdf/284/art%3A10.1007%2FBF02591962.pdf?originUrl=http://link.springer.com/article/10.1007/BF02591962&token2=exp=1484583587~acl=/static/pdf/284/art%253A10.1007%252FBF02591962.pdf?originUrl=http%3A%2F%2Flink.springer.com%2Farticle%2F10.1007%2FBF02591962*~hmac=d46ab943f1858d7bf70ab68270f1b3ba355551fc79047f2f0f6c6863c936c0a2>`__
        """
        qp_G = .5 * (P + P.T)   # quadprog assumes that P is symmetric
        qp_a = -q
        if A is not None:
            qp_C = -vstack([A, G]).T
            qp_b = -hstack([b, h])
            meq = A.shape[0]
        else:  # no equality constraint
            qp_C = -G.T
            qp_b = -h
            meq = 0
        return _quadprog_solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
except ImportError:
    print "\033[1;33m[pymanoid] Warning: quadprog solver not found\033[0;0m"
    quadprog_solve_qp = None


def cvxopt_solve_qp(P, q, G, h, A=None, b=None, solver=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} & & G x \leq h \\\\
            & & A x = b
        \\end{eqnarray}

    using CVXOPT
    <http://cvxopt.org/userguide/coneprog.html#quadratic-programming>.

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
        Set to 'mosek' to run MOSEK rather than CVXOPT.
    initvals : array, shape=(n,), optional
        Warm-start guess vector.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    # CVXOPT only considers the lower entries of P so we project on its
    # symmetric part beforehand
    P = .5 * (P + P.T)
    args = [cvxmat(P), cvxmat(q), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = cvxopt_qp(*args, solver=solver, initvals=initvals)
    if not ('optimal' in sol['status']):
        raise ValueError("QP optimum not found: %s" % sol['status'])
    return array(sol['x']).reshape((P.shape[1],))


try:
    import cvxopt.msk
    import mosek
    cvxopt.solvers.options['mosek'] = {mosek.iparam.log: 0}

    def mosek_solve_qp(P, q, G, h, A=None, b=None, initvals=None):
        return cvxopt_solve_qp(P, q, G, h, A, b, 'mosek', initvals)
except ImportError:
    print "\030[1;33m[pymanoid] Info: MOSEK solver not found\033[0;0m"


def solve_qp(P, q, G, h, A=None, b=None, solver=None):
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
    if quadprog_solve_qp is not None:
        return quadprog_solve_qp(P, q, G, h, A, b)
    return cvxopt_solve_qp(P, q, G, h, A, b)


def solve_safer_qp(P, q, G, h, w_reg, w_lin, solver='mosek'):
    """
    Solve the relaxed Quadratic Program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + (1/2) \\epsilon \\| s \\|^2 q^T x
        - w 1^T s\\\\
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
