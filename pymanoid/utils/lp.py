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
from cvxopt.solvers import lp, options
from numpy import array
from warnings import warn


options['show_progress'] = False  # disable cvxopt output


def solve_lp(c, G=None, h=None, A=None, b=None):
    args = [matrix(c)]
    if G is not None:
        args.extend([matrix(G), matrix(h)])
        if A is not None:
            args.extend([matrix(A), matrix(b)])
    sol = lp(*args)
    if not ('optimal' in sol['status']):
        warn("LP optimum not found: %s" % sol['status'])
        return None
    return array(sol['x']).reshape((c.shape[0],))
