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

from numpy import array, dot
from warnings import warn

solve_qp = None

try:
    from quadprog import solve_qp as _quadprog_solve_qp

    def quadprog_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:

            minimize
                (1/2) * x.T * P * x + q.T * x

            subject to
                G * x <= h
                A * x == b

        using quadprog <https://pypi.python.org/pypi/quadprog/>.
        """
        P = .5 * (P + P.T)  # quadprog assumes that P is symmetric
        return _quadprog_solve_qp(P, -q, -G.T, -h)[0]

    if solve_qp is None:
        solve_qp = quadprog_solve_qp
except ImportError:
    def quadprog_solve_qp(*args, **kwargs):
        raise ImportError("quadprog not found")

try:  # CVXOPT (2nd choice)
    from cvxopt import matrix
    from cvxopt.solvers import options, qp

    options['show_progress'] = False  # disable cvxopt output

    def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None, initvals=None):
        """
        Solve a Quadratic Program defined as:

            minimize
                (1/2) * x.T * P * x + q.T * x

            subject to
                G * x <= h
                A * x == b

        using CVXOPT <http://cvxopt.org/>.
        """
        n = P.shape[1]
        # CVXOPT 1.1.7 only considers the lower entries of P
        # so we need to project on the symmetric part beforehand,
        # otherwise a wrong cost function will be used
        P = .5 * (P + P.T)
        # now we can proceed
        args = [matrix(P), matrix(q)]
        if G is not None:
            args.extend([matrix(G), matrix(h)])
            if A is not None:
                args.extend([matrix(A), matrix(b)])
        sol = qp(*args, initvals=initvals)
        if not ('optimal' in sol['status']):
            warn("QP optimum not found: %s" % sol['status'])
            return None
        return array(sol['x']).reshape((n,))

    if solve_qp is None:
        solve_qp = cvxopt_solve_qp
except ImportError:
    def cvxopt_solve_qp(*args, **kwargs):
        raise ImportError("CVXOPT not found")


def solve_relaxed_qp(P, q, G, h, A, b, tol=None, OVER_WEIGHT=100000.):
    """
    Solve a relaxed version of the Quadratic Program:

        min_x   x.T * P * x + 2 * q.T * x

         s.t.   G * x <= h
                A * x == b

    The relaxed problem is defined by

        min_x   c1(x, P, q) + OVER_WEIGHT * c2(x, A, b)
         s.t.   G * x <= h

    where c1(x, P, q) is the initial cost, OVER_WEIGHT is a very high weight and

        c1(x, P, q) = x.T * P * x + 2 * q.T * x
        c2(x, A, b) = |A * x - b|^2

    If ``tol`` is not None, the solution will only be returned if the relative
    variation between A * x and b is less than ``tol``.
    """
    P2 = P + OVER_WEIGHT * dot(A.T, A)
    q2 = q + OVER_WEIGHT * dot(-b.T, A)
    x = solve_qp(P2, q2, G, h)
    if tol is not None:
        def sq(v):
            return dot(v, v)
        if sq(dot(A, x) - b) / sq(b) > tol * tol:
            return None
    return x
