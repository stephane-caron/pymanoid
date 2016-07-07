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


from qpoases import PyQProblem as QProblem
from qpoases import PyQProblemB as QProblemB
from qpoases import PyOptions as Options
from qpoases import PyPrintLevel as PrintLevel
from numpy import array, hstack, ones, vstack, zeros


__infty = 1e10
options = Options()
options.printLevel = PrintLevel.NONE
nb_wsr = array([10])  # number of working set recalculations


def solve_qp(P, q, G=None, h=None, A=None, b=None, warm_start=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            1/2 * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    """
    n = P.shape[0]
    lb = -__infty * ones(P.shape[0])
    ub = +__infty * ones(P.shape[0])
    has_cons = G is not None or A is not None
    if G is not None and A is None:
        C = G
        lb_C = -__infty * ones(h.shape[0])
        ub_C = h
    elif G is None and A is not None:
        C = vstack([A, A])
        lb_C = h
        ub_C = h
    else:  # G is not None and A is not None:
        C = vstack([G, A, A])
        lb_C = hstack([-__infty * ones(h.shape[0]), b])
        ub_C = hstack([h, b])
    if has_cons:
        qp = QProblem(n, C.shape[0])
        qp.setOptions(options)
        qp.init(P, q, C, lb, ub, lb_C, ub_C, nb_wsr)
    else:
        qp = QProblemB(n)
        qp.setOptions(options)
        qp.init(P, q, lb, ub, nb_wsr)
    x_opt = zeros(n)
    qp.getPrimalSolution(x_opt)
    return x_opt
