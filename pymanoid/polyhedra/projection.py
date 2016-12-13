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

import bretl
import cdd
import cvxopt

from numpy import array, dot, hstack, zeros


class PolyhedronProjector(object):

    def __init__(self, A=None, b=None, C=None, d=None, E=None, f=None,
                 box_size=42):
        """
        Project a polytope defined by:

            A * x <= b
            C * x == d

        While the output (projection) is computed from:

            y = E * x + f

        INPUT:

        - ``A`` -- inequality matrix
        - ``b`` -- inequality vector
        - ``C`` -- equality matrix
        - ``d`` -- equality vector
        - ``E`` -- output matrix
        - ``f`` -- output vector
        - ``box_size`` -- used to make sure the output is a polygon
        """
        self.A = A
        self.C = C
        self.E = E
        self.b = b
        self.box_size = box_size
        self.d = d
        self.f = f

    def set_inequality(self, A, b):
        """
        Set the inequality constraint ``A * x <= b`` of the polytope.
        """
        self.A = A
        self.b = b

    def set_equality(self, C, d):
        """
        Set the equality constraint ``C * x == d`` of the polytope.
        """
        self.C = C
        self.d = d

    def set_output(self, E, f):
        """
        Set the output definition ``y = E * x + d`` of the projected polytope.
        """
        self.E = E
        self.f = f

    def _prepare(self):
        assert self.A is not None and self.b is not None
        assert self.C is not None and self.d is not None
        assert self.E is not None and self.f is not None
        return self.A, self.b, self.C, self.d, self.E, self.f

    def project(self):
        """
        Project polyhedron.

        OUTPUT:

        Pair ``(v, r)`` where ``v`` is the list of vertices and ``r`` the list
        of rays (should be empty) of the projected polytope.
        """
        return self.project_cdd()

    def project_cdd(self):
        """
        Project polytope using the double-description method.

        OUTPUT:

        Pair ``(v, r)`` where ``v`` is the list of vertices and ``r`` the list
        of rays of the projected polyhedron.
        """
        A, b, C, d, E, f = self._prepare()
        b = b.reshape((b.shape[0], 1))

        # the input [b, -A] to cdd.Matrix represents (b - A * x >= 0)
        # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
        linsys = cdd.Matrix(hstack([b, -A]), number_type='float')
        linsys.rep_type = cdd.RepType.INEQUALITY

        # the input [d, -C] to cdd.Matrix.extend represents (d - C * x == 0)
        # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
        d = d.reshape((d.shape[0], 1))
        linsys.extend(hstack([d, -C]), linear=True)

        # Convert from H- to V-representation
        linsys.canonicalize()
        P = cdd.Polyhedron(linsys)
        generators = P.get_generators()
        if generators.lin_set:
            print "Generators have linear set:", generators.lin_set
        V = array(generators)

        # Project output wrenches to 2D set
        vertices, rays = [], []
        free_coordinates = []
        for i in xrange(V.shape[0]):
            if generators.lin_set and i in generators.lin_set:
                free_coordinates.append(list(V[i, 1:]).index(1.))
            elif V[i, 0] == 1:  # vertex
                vertices.append(dot(E, V[i, 1:]) + f)
            else:  # ray
                rays.append(dot(E, V[i, 1:]))
        return vertices, rays


class PolytopeProjector(PolyhedronProjector):

    def project(self, method='bretl'):
        """
        Project polytope.

        INPUT:

        - ``method`` -- algorithm to use, to choose between 'bretl' and 'cdd'

        OUTPUT:

        List of vertices.
        """
        assert method in ['bretl', 'cdd']
        if method == 'cdd':
            vertices, rays = self.project_cdd()
            assert not rays, "This polyhedron is not a polytope"
            return vertices
        return self.project_bretl()

    def project_bretl(self, solver='glpk'):
        """
        Project polytope using the incremental projection algorithm
        from [BL08].

        INPUT:

        - ``solver`` -- (optional) LP backend for CVXOPT, 'glpk' is recommended

        OUTPUT:

        List of vertices.

        REFERENCES:

        .. [BL08] http://dx.doi.org/10.1109/TRO.2008.2001360
        """
        A, b, C, d, E, f = self._prepare()

        # Inequality constraints: A_ext * [ x  u  v ] <= b_ext iff
        # (1) A * x <= b and (2) |u|, |v| <= box_size
        A_ext = zeros((A.shape[0] + 4, A.shape[1] + 2))
        A_ext[:-4, :-2] = A
        A_ext[-4, -2] = 1
        A_ext[-3, -2] = -1
        A_ext[-2, -1] = 1
        A_ext[-1, -1] = -1
        A_ext = cvxopt.matrix(A_ext)

        b_ext = zeros(b.shape[0] + 4)
        b_ext[:-4] = b
        b_ext[-4:] = array([self.box_size] * 4)
        b_ext = cvxopt.matrix(b_ext)

        # Equality constraints: C_ext * [ x  u  v ] == d_ext iff
        # (1) C * x == d and (2) [ u  v ] == E * x + f
        C_ext = zeros((C.shape[0] + 2, C.shape[1] + 2))
        C_ext[:-2, :-2] = C
        C_ext[-2:, :-2] = E[:2]
        C_ext[-2:, -2:] = array([[-1, 0], [0, -1]])
        C_ext = cvxopt.matrix(C_ext)

        d_ext = zeros(d.shape[0] + 2)
        d_ext[:-2] = d
        d_ext[-2:] = -f[:2]
        d_ext = cvxopt.matrix(d_ext)

        lp_obj = cvxopt.matrix(zeros(A.shape[1] + 2))
        lp = lp_obj, A_ext, b_ext, C_ext, d_ext
        res, P = bretl.ComputePolygon(lp, solver=solver)
        if not res:
            msg = "bretl.ComputePolygon: "
            msg += "could not optimize in direction %s" % str(P)
            raise Exception(msg)

        P.sort_vertices()
        vertices_list = P.export_vertices()
        vertices = [array([v.x, v.y]) for v in vertices_list]
        return vertices
