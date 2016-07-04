#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
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


from cvxopt import matrix
from cvxopt.solvers import options, qp
from numpy import array, dot
from vectors import norm
from warnings import warn


options['show_progress'] = False  # disable cvxopt output


def solve_qp(P, q, G=None, h=None, A=None, b=None):
    """
    Solve a Quadratic Program defined by

        min_x   x.T * P * x + 2 * q.T * x

         s.t.   G * x <= h
                A * x == b

    """
    P_sym = .5 * (P + P.T)   # necessary for CVXOPT 1.1.7
    #
    # CVXOPT 1.1.7 only considers the lower entries of P
    # so we need to project on the symmetric part beforehand,
    # otherwise a wrong cost function will be used
    #
    args = [matrix(P_sym), matrix(q)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = qp(*args)
    if not ('optimal' in sol['status']):
        warn("QP optimum not found: %s" % sol['status'])
        return None
    return array(sol['x']).reshape((P.shape[1],))


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
    if tol is not None and norm(dot(A, x) - b) / norm(b) > tol:
        return None
    return x
