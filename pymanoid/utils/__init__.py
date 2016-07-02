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


import math
import numpy

from cvxopt_ import cvxopt_solve_lp
from cvxopt_ import cvxopt_solve_qp
from matplotlib_ import plot_polygon


def norm(v):
    """
    For some reason, numpy's one is slow. On my machine:

        In [1]: %timeit numpy.linalg.norm(v)
        100000 loops, best of 3: 3.9 µs per loop

        In [2]: %timeit pymanoid.utils.norm(v)
        1000000 loops, best of 3: 727 ns per loop

    """
    return math.sqrt(numpy.dot(v, v))


def normalize(v):
    """Return a unit vector u such that v = norm(v) * u."""
    return v / norm(v)


solve_lp = cvxopt_solve_lp
solve_qp = cvxopt_solve_qp

__all__ = [
    'cvxopt_solve_lp',
    'cvxopt_solve_qp',
    'norm',
    'plot_polygon'
    'solve_lp',
    'solve_qp',
]
