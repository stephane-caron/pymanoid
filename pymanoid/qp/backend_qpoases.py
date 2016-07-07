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
from numpy import array, hstack, ones, vstack, zeros


def solve_qp(P, q, G=None, h=None, A=None, b=None, warm_start=None):
    """
    Solve a Quadratic Program defined as:

        minimize
            1/2 * x.T * P * x + q.T * x

        subject to
            G * x <= h
            A * x == b

    """

    # min   1/2*x'Hx + x'g
    # s.t.  lb  <=  x <= ub
    #       lbA <= Ax <= ubA
    nb_wsr = array([10])  # number of working set recalculations
    if G is None and A is None or True:
        qp = QProblemB(P.shape[0])
        qp.init(P, q, None, None, nb_wsr)
    else:
        infty = 1e10
        if G is not None and A is None:
            C, lb_C, ub_C = G, -infty * ones(h.shape[0]), h
        elif G is None and A is not None:
            C, lb_C, ub_C = vstack([A, A]), h, h
        else:  # G is not None and A is not None
            C = vstack([G, A, A])
            lb_C = hstack([-infty * ones(C.shape[0]), b])
            ub_C = hstack([h, b])
        # options = Options()
        # options.printLevel = PrintLevel.NONE
        # example.setOptions(options)
        qp = QProblem(P.shape[0], C.shape[0])
        qp.init(P, q, C, None, None, lb_C, ub_C, nb_wsr)

    x_opt = zeros(P.shape[0])
    qp.getPrimalSolution(x_opt)
    return x_opt
