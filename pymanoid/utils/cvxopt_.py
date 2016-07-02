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


import cvxopt
import cvxopt.solvers

from exceptions import OptimalNotFound
from numpy import array

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output


def cvxopt_solve_lp(c, G=None, h=None, A=None, b=None):
    args = [cvxopt.matrix(c)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.lp(*args)
    if not ('optimal' in sol['status']):
        raise OptimalNotFound(sol['status'])
    return array(sol['x']).reshape((c.shape[0],))


def cvxopt_solve_qp(P, q, G=None, h=None, A=None, b=None):
    P_sym = .5 * (P + P.T)   # necessary for CVXOPT 1.1.7
    #
    # CVXOPT 1.1.7 only considers the lower entries of P
    # so we need to project on the symmetric part beforehand,
    # otherwise a wrong cost function will be used
    #
    args = [cvxopt.matrix(P_sym), cvxopt.matrix(q)]
    if G is not None:
        args.extend([cvxopt.matrix(G), cvxopt.matrix(h)])
        if A is not None:
            args.extend([cvxopt.matrix(A), cvxopt.matrix(b)])
    sol = cvxopt.solvers.qp(*args)
    if not ('optimal' in sol['status']):
        raise OptimalNotFound(sol['status'])
    return array(sol['x']).reshape((P.shape[1],))
