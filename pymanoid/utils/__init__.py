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


from lp import solve_lp
from qp import solve_qp, solve_relaxed_qp
from matplotlib_ import plot_polygon
from numpy import dot, eye, zeros
from vectors import norm, normalize


def is_positive_combination(b, A):
    """
    Check if b can be written as a positive combination of lines from A.

    INPUT:

    - ``b`` -- test vector
    - ``A`` -- matrix of line vectors to combine

    OUTPUT:

    True if and only if b = A.T * x for some x >= 0.
    """
    m = A.shape[0]
    P, q = eye(m), zeros(m)
    #
    # NB: one could try solving a QP minimizing |A * x - b|^2 (and no equality
    # constraint), however the precision of the output is quite low (~1e-1).
    #
    G, h = -eye(m), zeros(m)
    x = solve_qp(P, q, G, h, A.T, b)
    if x is None:  # optimum not found
        return False
    return norm(dot(A.T, x) - b) < 1e-10 and min(x) > -1e-10


__all__ = [
    'norm',
    'normalize',
    'plot_polygon',
    'solve_lp',
    'solve_qp',
    'solve_relaxed_qp',
]
