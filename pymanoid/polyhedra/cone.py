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

from numpy import array, average, dot, hstack, sqrt, vstack, zeros

from draw import draw_polygon, matplotlib_to_rgba
from polyhedron import Polyhedron
from sim import get_openrave_env


def norm(v):
    return sqrt(dot(v, v))


class Cone(Polyhedron):

    """
    Cones are polyhedra with only rays and a single apex at the origin.

    H-rep: the matrix A such that the cone is defined by A * x <= 0
    V-rep: the set {r_1, ..., r_k} of ray vectors such that the cone is defined
           by x = nonneg(r_1, ..., r_k)
    """

    def __init__(self, face=None, rays=None):
        assert rays is not None or face is not None, \
            "needs either the face matrix or a set of rays"
        hrep = None
        vrep = None
        if rays is not None:
            R = array(rays)
            vrep = vstack([
                hstack([zeros((R.shape[0], 1)), R]),
                hstack([[1], zeros(R.shape[1])])])
        if face is not None:
            hrep = hstack([zeros((face.shape[0], 1)), -face])
        self.__face = face
        self.__rays = rays
        super(Cone, self).__init__(hrep, vrep)

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
        Rays, also known as positive generators of the cone.

        .. NOTE::

            Steel on the skyline
            Sky made of glass
            Made for a real world
            All things must pass
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

    def span(self):
        """
        Span matrix S such that the cone is defined by x = S * z (z >= 0).
        """
        return array(self.rays()).T

    """
    Backward compatibility
    ======================
    """

    @staticmethod
    def span_of_face(F):
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
    def face_of_span(S):
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


class Cone3D(Cone):

    def draw(self, apex, size=1., combined='g-#', color=None, linewidth=2):
        """
        Draw cone with apex at a given world position.

        INPUT:

        - ``apex`` -- position of the apex of the cone in world coordinates
        - ``size`` -- scale factor (default: 1.)
        - ``combined`` -- drawing spec in matplotlib fashion (default: 'g-#')
        - ``color`` -- color letter or RGBA tuple
        - ``linewidth`` -- thickness of the edges of the cone

        OUTPUT:

        A list of OpenRAVE handles. Must be stored in some variable, otherwise
        the drawn object will vanish instantly.
        """
        assert len(apex) == 3, "apex is not a 3D point"
        rays = self.rays()
        if color is None:
            color = matplotlib_to_rgba(combined[0])
        if type(color) is str:
            color = matplotlib_to_rgba(color)
        env = get_openrave_env()
        normal = average(rays, axis=0)
        normal /= norm(normal)
        section = [apex + ray * size / dot(normal, ray) for ray in rays]
        handles = draw_polygon(
            points=section, normal=normal, combined=combined, color=color)
        edges = vstack([[apex, vertex] for vertex in section])
        edge_color = array(color) * 0.7
        edge_color[3] = 1.
        handles.append(env.drawlinelist(
            edges, linewidth=linewidth, colors=edge_color))
        return handles
