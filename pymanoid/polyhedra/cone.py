#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
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

    Parameters
    ----------
    face : ndarray
        Face matrix :math:`A` of the polyhedral cone.
    rays : list of ndarrays
        List of rays definied the V-representation of the cone.

    Notes
    -----

    By the `Minkowski-Weyl theorem
    <https://www.inf.ethz.ch/personal/fukudak/polyfaq/node14.html>`_,
    polyhedral convex cones can be described equivalently in:

    - *H-representation*: a matrix :math:`A` such that the cone is defined by

    .. math::

        A x \leq 0

    - *V-representation*: a set :math:`\\{r_1, \\ldots, r_k\\}` of ray vectors
      such that the cone is defined by

    .. math::

        x = \\mathrm{nonneg}(r_1, \\ldots, r_k)
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
        self.face = face
        self.rays = rays
        super(Cone, self).__init__(hrep, vrep)

    def compute_face(self):
        """
        Face matrix F such that the cone is defined by F * x <= 0.
        """
        if self.face is not None:
            return self.face
        bA = self.hrep
        for h in bA:
            if norm(h[1:]) < 1e-10:
                continue
            elif abs(h[0]) > 1e-10:  # should be zero for a cone
                raise TypeError("Polyhedron is not a cone")
        self.face = -bA[:, 1:]  # H-rep is [b | A] == [0 | -F]
        return self.face

    def compute_rays(self):
        """
        Rays, also known as positive generators of the cone.

        .. NOTE::

            Steel on the skyline
            Sky made of glass
            Made for a real world
            All things must pass
        """
        if self.rays is not None:
            return self.rays
        tV = self.vrep()  # V-rep is [t | V]
        for i in xrange(tV.shape[0]):
            if tV[i, 0] != 0:  # t = 1 for vertex, 0 for ray
                raise Exception("Polyhedron is not a cone")
        self.rays = list(tV[:, 1:])
        return self.rays

    def span(self):
        """
        Span matrix S such that the cone is defined by x = S * z (z >= 0).
        """
        if self.rays is None:
            return None
        return array(self.rays).T

    def draw(self, apex, size=1., combined='g-#', color=None, linewidth=2):
        """
        Draw cone with apex at a given world position.

        Parameters
        ----------
        apex : array
            Position of the origin of the cone in world coordinates.
        size : scalar, optional
            Scale factor.
        combined : string, optional
            Drawing spec in matplotlib fashion. Default is 'g-#'.
        color : char or triplet, optional
            Color letter or RGB values, default is 'g' for green.
        linewidth : scalar
            Thickness of drawn line.

        Returns
        -------
        handles : list of openravepy.GraphHandle
            OpenRAVE graphical handles. Must be stored in some variable,
            otherwise the drawn object will vanish instantly.
        """
        assert len(apex) == 3, "apex is not a 3D point"
        assert len(self.rays[0]) == 3, "only 3D cones can be drawn"
        rays = self.rays
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

    """
    Backward compatibility
    ======================
    """

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
