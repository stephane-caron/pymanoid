#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
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
import os
import sys

from numpy import array, dot, hstack, ones, vstack, sqrt, zeros

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/..')

from optim import solve_lp
from polyhedron import Polyhedron


def norm(v):
    return sqrt(dot(v, v))


class EmptyPolytope(Exception):

    pass


class Polytope(Polyhedron):

    """Polytopes are bounded polyhedra, i.e., with only vertices and no ray."""

    def __init__(self, hrep=None, vrep=None, vertices=None):
        if vrep is None:
            if vertices is not None:
                V = vstack(vertices)
                t = ones((V.shape[0], 1))  # first column is 1 for vertices
                vrep = hstack([t, V])
        super(Polytope, self).__init__(hrep, vrep)

    @staticmethod
    def hrep(vertices):
        """
        Compute the half-space representation (A, b) of a polytope defined as
        convex hull of a set of vertices, that is:

            {A * x <= b}  if and only if  {x \in conv(vertices)}

        """
        V = vstack(vertices)
        t = ones((V.shape[0], 1))  # first column is 1 for vertices
        tV = hstack([t, V])
        mat = cdd.Matrix(tV, number_type='float')
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(mat)
        bA = array(P.get_inequalities())
        if bA.shape == (0,):  # bA == []
            return bA
        # the polyhedron is given by b + A x >= 0 where bA = [b|A]
        b, A = array(bA[:, 0]), -array(bA[:, 1:])
        return (A, b)

    @staticmethod
    def vrep(A, b):
        """
        Compute the vertices of a polytope given in half-space representation by

            A * x <= b

        """
        b = b.reshape((b.shape[0], 1))
        mat = cdd.Matrix(hstack([b, -A]), number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        g = P.get_generators()
        V = array(g)
        vertices = []
        for i in xrange(V.shape[0]):
            if V[i, 0] != 1:  # 1 = vertex, 0 = ray
                raise Exception("Polyhedron is not a polytope")
            elif i not in g.lin_set:
                vertices.append(V[i, 1:])
        return vertices

    @staticmethod
    def compute_chebyshev_center(A, b):
        """
        Compute the Chebyshev center of a polyhedron, that is, the point
        furthest away from all inequalities.

        INPUT:

        - ``A`` -- matrix of polytope H-representation
        - ``b`` -- vector of polytope H-representation

        OUTPUT:

        A numpy array of shape ``(A.shape[1],)``.

        REFERENCES:

        Stephen Boyd and Lieven Vandenberghe, "Convex Optimization",
        Section 4.3.1, p. 148.
        """
        cost = zeros(A.shape[1] + 1)
        cost[-1] = -1.
        a_cheby = array([norm(A[i, :]) for i in xrange(A.shape[0])])
        A_cheby = hstack([A, a_cheby.reshape((A.shape[0], 1))])
        z = solve_lp(cost, A_cheby, b)
        if z[-1] < -1e-1:  # last coordinate is distance to boundaries
            raise EmptyPolytope("Chebyshev center violation by %.2f" % z[-1])
        return z[:-1]
