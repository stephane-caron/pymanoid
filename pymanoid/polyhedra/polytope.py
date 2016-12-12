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

from numpy import array, dot, eye, hstack, ones, vstack, sqrt, zeros

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/..')

from optim import solve_lp, solve_qp
from polyhedron import Polyhedron


def norm(v):
    return sqrt(dot(v, v))


class Polytope(Polyhedron):

    """Polytopes are bounded polyhedra, i.e., with only vertices and no ray."""

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
        assert z[-1] > 0  # last coordinate is distance to boundaries
        return z[:-1]


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


def is_redundant(vectors):
    """
    Check if a set of vectors is redundant, i.e. one of them can be written as
    positive combination of the others.

    .. NOTE::

        When using cvxopt, this function may print out a significant number
        of messages "Terminated (singular KKT matrix)." in the terminal.

    """
    F = array(vectors)
    all_lines = set(range(F.shape[0]))
    return any([is_positive_combination(
        F[i], F[list(all_lines - set([i]))]) for i in all_lines])
