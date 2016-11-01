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

from misc import norm
from numpy import array, dot, eye, hstack, ones, vstack, zeros
from optim import solve_lp, solve_qp


class Polyhedron(object):

    """
    Wrapper for the CDD library.

    Computes both the H-rep (halfspace repsentation) and V-rep (vertex/rays
    representation) of a polyhedron, in matrix form:

    - H-rep: [b | A] where the polyhedron is defined by b + A * x >= 0
    - V-rep: [t | V] where V stacks vertices as row vectors and t is the type
                     vector (1 for points, 0 for rays/lines)

    See also: <https://github.com/haudren/pyparma>
    """

    number_type = 'float'

    def __init__(self, hrep=None, vrep=None):
        assert hrep is not None or vrep is not None, \
            "Please provide either H-rep or V-rep."
        self.__hrep = hrep
        self.__vrep = vrep

    def hrep(self):
        if self.__hrep is not None:
            return self.__hrep
        mat = cdd.Matrix(self.vrep, number_type=self.number_type)
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(mat)
        ineq = P.get_inequalities()
        if ineq.lin_set:
            raise NotImplementedError("polyhedron has linear generators")
        self.__hrep = array(ineq)
        return self.__hrep

    def vrep(self):
        if self.__vrep is not None:
            return self.__vrep
        mat = cdd.Matrix(self.hrep, number_type=self.number_type)
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        gen = P.get_generators()
        if gen.lin_set:
            raise NotImplementedError("polyhedron has linear generators")
        self.__vrep = array(gen)
        return self.__vrep


class Cone(Polyhedron):

    """
    Cones are polyhedra with only rays and a single apex at the origin.

    H-rep: the matrix A such that the cone is defined by A * x <= 0
    V-rep: the set {r_1, ..., r_k} of ray vectors such that the cone is defined
           by x = nonneg(r_1, ..., r_k)
    """

    def __init__(self, face=None, rays=None):
        super(Cone, self).__init__(hrep=None, vrep=None)
        if rays is not None:
            R = array(rays)
            self.__rays = rays
            self.__vrep = vstack([
                hstack([zeros((R.shape[0], 1)), R]),
                hstack([1, zeros(R.shape[0])])])
        elif face is not None:
            self.__face = face
            self.__hrep = hstack([zeros((face.shape[0], 1)), -face])
        else:
            raise ValueError("needs either the face matrix or a set of rays")

    def face(self):
        """
        Face matrix F such that the cone is defined by F * x <= 0.
        """
        if self.__face is not None:
            return self.__face
        # A = []
        H = self.hrep()  # H-rep is [b | -A]
        for h in H:
            if norm(h[1:]) < 1e-10:
                continue
            elif abs(h[0]) > 1e-10:  # should be zero for a cone
                raise TypeError("Polyhedron is not a cone")
            # elif i in ineq.lin_set:
            #     raise TypeError("Polyhedron has linear generators")
            # A.append(-H[i, 1:])
        self.__face = -H[:, 1:]
        return self.__face

    def rays(self):
        """
        Rays {r_1, ..., r_k} such that the cone is defined by
        x = nonneg(r_1, ..., r_k).
        """
        if self.__rays is not None:
            return self.__rays
        V = self.vrep()
        # rays = []
        for i in xrange(V.shape[0]):
            if V[i, 0] != 0:  # 1 = vertex, 0 = ray
                raise Exception("Polyhedron is not a cone")
            # elif i not in g.lin_set:  # ignore those in lin_set
            #     rays.append(V[i, 1:])
        self.__rays = list(V[:, 1:])
        return self.__rays

    """
    Backward compatibility
    ======================
    """

    @staticmethod
    def span_of_face(F):  # TODO: remove
        b, A = zeros((F.shape[0], 1)), F
        # H-representation: b - A x >= 0
        # ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
        # the input to pycddlib is [b, -A]
        mat = cdd.Matrix(hstack([b, -A]), number_type='float')
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        g = P.get_generators()
        V = array(g)
        rays = []
        for i in xrange(V.shape[0]):
            if V[i, 0] != 0:  # 1 = vertex, 0 = ray
                raise Exception("Polyhedron is not a cone")
            elif i not in g.lin_set:  # ignore those in lin_set
                rays.append(V[i, 1:])
        return array(rays).T

    @staticmethod
    def face_of_span(S):  # TODO: remove
        V = vstack([
            hstack([zeros((S.shape[1], 1)), S.T]),
            hstack([1, zeros(S.shape[0])])])
        # V-representation: first column is 0 for rays
        mat = cdd.Matrix(V, number_type='float')
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(mat)
        ineq = P.get_inequalities()
        H = array(ineq)
        if H.shape == (0,):  # H == []
            return H
        A = []
        for i in xrange(H.shape[0]):
            # H matrix is [b, -A] for A * x <= b
            if norm(H[i, 1:]) < 1e-10:
                continue
            elif abs(H[i, 0]) > 1e-10:  # b should be zero for a cone
                raise Exception("Polyhedron is not a cone")
            elif i not in ineq.lin_set:
                A.append(-H[i, 1:])
        return array(A)


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
