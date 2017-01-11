#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
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

import cvxopt
import cvxopt.solvers

from cvxopt import matrix as cvxmat
from cvxopt.solvers import lp as cvxopt_lp
from cvxopt.solvers import qp as cvxopt_qp
from numpy import array, dot
from warnings import warn

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
    LP_SOLVER = None


def solve_lp(c, G, h, A=None, b=None, solver=LP_SOLVER):
    """
    Solve a Linear Program defined by:

        minimize
            c.T * x

        subject to
            G * x <= h
            A * x == b  (optional)

    using CVXOPT <http://cvxopt.org/userguide/coneprog.html#linear-programming>.

    INPUT:

    - ``c`` -- cost vector
    - ``G`` -- inequality matrix
    - ``h`` -- inequality vector
    - ``A`` -- (optional) equality matrix
    - ``b`` -- (optional) equality vector
    - ``solver`` -- (optional) solver to use, defaults to GLPK if available
    """
    args = [cvxmat(c), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = cvxopt_lp(*args, solver=solver)
    if 'optimal' not in sol['status']:
        return None
    return array(sol['x']).reshape((c.shape[0],))


"""
Quadratic Programming
=====================
"""

try:
    # quadprog is the fastest QP solver I could find so far
    from quadprog import solve_qp as _quadprog_solve_qp

    def quadprog_solve_qp(P, q, G, h):
        """
        Solve a Quadratic Program defined as:

            minimize
                (1/2) * x.T * P * x + q.T * x

            subject to
                G * x <= h

        using quadprog <https://pypi.python.org/pypi/quadprog/>.

        INPUT:

        - ``P`` -- primal quadratic cost matrix
        - ``q`` -- primal quadratic cost vector
        - ``G`` -- linear inequality constraint matrix
        - ``h`` -- linear inequality constraint vector

        OUTPUT:

        A numpy.array with the solution ``x``, if found, otherwise None.
        """
        # quadprog assumes that P is symmetric so we project it and its
        # symmetric part beforehand
        P = .5 * (P + P.T)
        return _quadprog_solve_qp(P, -q, -G.T, -h)[0]
except ImportError:
    warn("QP solver: quadprog not found, falling back to CVXOPT")
    quadprog_solve_qp = None


def cvxopt_solve_qp(P, q, G, h, A=None, b=None, solver=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            (1/2) * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b  (optional)

    using CVXOPT
    <http://cvxopt.org/userguide/coneprog.html#quadratic-programming>.

    INPUT:

    - ``P`` -- primal quadratic cost matrix
    - ``q`` -- primal quadratic cost vector
    - ``G`` -- linear inequality constraint matrix
    - ``h`` -- linear inequality constraint vector
    - ``A`` -- (optional) linear equality constraint matrix
    - ``b`` -- (optional) linear equality constraint vector
    - ``solver`` -- (optional) use 'mosek' to run MOSEK rather than CVXOPT
    - ``initvals`` -- (optional) warm-start guess

    OUTPUT:

    A numpy.array with the solution ``x``, if found, otherwise None.
    """
    # CVXOPT only considers the lower entries of P so we project on its
    # symmetric part beforehand
    P = .5 * (P + P.T)
    args = [cvxmat(P), cvxmat(q), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = cvxopt_qp(*args, solver=solver, initvals=initvals)
    if not ('optimal' in sol['status']):
        warn("QP optimum not found: %s" % sol['status'])
        return None
    return array(sol['x']).reshape((P.shape[1],))


try:
    import cvxopt.msk
    import mosek
    cvxopt.solvers.options['mosek'] = {mosek.iparam.log: 0}

    def mosek_solve_qp(P, q, G, h, A=None, b=None, initvals=None):
        return cvxopt_solve_qp(P, q, G, h, A, b, 'mosek', initvals)
except ImportError:
    pass


if quadprog_solve_qp is not None:
    solve_qp = quadprog_solve_qp
else:  # fallback option is CVXOPT
    solve_qp = cvxopt_solve_qp


def solve_relaxed_qp(P, q, G, h, A, b, tol=None, OVER_WEIGHT=100000.):
    """
    Solve a relaxed version of the Quadratic Program:

        min_x   c1(x)

         s.t.   G * x <= h
                c2(x) == 0

    where the cost function and linear equalities are given by:

        c1(x) = x.T * P * x + 2 * q.T * x
        c2(x) = |A * x - b|^2

    The relaxed problem is defined by

        min_x   c1(x, P, q) + OVER_WEIGHT * c2(x, A, b)
         s.t.   G * x <= h

    where OVER_WEIGHT is a very high weight.

    If ``tol`` is not None, the solution will only be returned if the relative
    variation between A * x and b is less than ``tol``.
    """
    P2 = P + OVER_WEIGHT * dot(A.T, A)
    q2 = q + OVER_WEIGHT * dot(-b.T, A)
    x = solve_qp(P2, q2, G, h)
    if x is not None and tol is not None:
        def sq(v):
            return dot(v, v)
        if sq(dot(A, x) - b) / sq(b) > tol * tol:
            return None
    return x
