#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
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


from numpy import dot


try:  # CVXOPT
    from backend_cvxopt import cvxopt_solve_qp
except ImportError:
    def cvxopt_solve_qp(*args, **kwargs):
        raise ImportError("CVXOPT not found")

try:  # quadprog
    from backend_quadprog import quadprog_solve_qp
except ImportError:
    def quadprog_solve_qp(*args, **kwargs):
        raise ImportError("quadprog not found")


try:  # qpOASES
    from backend_qpoases import qpoases_solve_qp
except ImportError:
    def qpoases_solve_qp(*args, **kwargs):
        raise ImportError("qpOASES not found")


solve_qp = cvxopt_solve_qp


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


__all__ = [
    'cvxopt_solve_qp',
    'qpoases_solve_qp',
    'quadprog_solve_qp',
]
