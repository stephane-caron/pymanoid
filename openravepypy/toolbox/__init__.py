#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of openravepypy.
#
# openravepypy is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# openravepypy is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openravepypy. If not, see <http://www.gnu.org/licenses/>.


from cvxopt_wrapper import cvxopt_solve_qp
from numpy import dot, sqrt


def norm(v):
    """
    For some reason, pylab's one is slow. On my machine:

        In [1]: from pylab import norm as pynorm

        In [5]: %timeit pynorm(v)
        100000 loops, best of 3: 10.3 µs per loop

        In [11]: %timeit norm(v)
        100000 loops, best of 3: 4.82 µs per loop

    """
    return sqrt(dot(v, v))
