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


import cdd
import numpy

from toolbox import norm


class NotConeFace(Exception):

    def __init__(self, F, V):
        self.F = F
        self.V = V

    def __str__(self):
        return "Matrix F is not a cone face"


class NotConeSpan(Exception):

    def __init__(self, S, H, i):
        self.S = S
        self.H = H
        self.i = i

    def __str__(self):
        return "Matrix S is not a cone span"


def span_of_face(F):
    """
    Compute the span matrix S of the face matrix F, which is such that

        {F x <= 0}  if and only if  {x = F^S z, z >= 0}

    """
    b, A = numpy.zeros((F.shape[0], 1)), F
    # H-representation: b - A x >= 0
    # ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    # the input to pycddlib is [b, -A]
    F_cdd = cdd.Matrix(numpy.hstack([b, -A]), number_type='float')
    F_cdd.rep_type = cdd.RepType.INEQUALITY
    P = cdd.Polyhedron(F_cdd)
    g = P.get_generators()
    V = numpy.array(g)
    rays = []
    for i in xrange(V.shape[0]):
        if V[i, 0] != 0:  # 1 = vertex, 0 = ray
            raise NotConeFace(F, V)
        elif i not in g.lin_set:
            rays.append(V[i, 1:])
    return numpy.array(rays).T


def face_of_span(S):
    """
    Compute the face matrix F of the span matrix S, which is such that

        {x = S z, z >= 0}  if and only if  {F x <= 0}.

    """
    V = numpy.vstack([
        numpy.hstack([numpy.zeros((S.shape[1], 1)), S.T]),
        numpy.hstack([1, numpy.zeros(S.shape[0])])])
    # V-representation: first column is 0 for rays
    V_cdd = cdd.Matrix(V, number_type='float')
    V_cdd.rep_type = cdd.RepType.GENERATOR
    P = cdd.Polyhedron(V_cdd)
    ineq = P.get_inequalities()
    H = numpy.array(ineq)
    if H.shape == (0,):  # H = []
        return H
    A = []
    for i in xrange(H.shape[0]):
        # H matrix is [b, -A] for A * x <= b
        if norm(H[i, 1:]) < 1e-10:
            continue
        elif abs(H[i, 0]) > 1e-10:  # b should be zero for a cone
            raise NotConeSpan(S, H, i)
        elif i not in ineq.lin_set:
            A.append(-H[i, 1:])
    return numpy.array(A)
