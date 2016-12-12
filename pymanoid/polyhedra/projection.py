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


def project_polytope_bretl(A, b, C, d, E, f, box_size=42, solver='glpk'):
    """
    Project a polytope using an incremental projection algorithm.

    The initial polytope is defined by:

        A * x <= b
        C * x == d

    While the output (projection) is computed from:

        y = E * x + f

    For details, see <http://dx.doi.org/10.1109/TRO.2008.2001360> and
    <https://hal.archives-ouvertes.fr/hal-01201060/>.
    """
    # Inequality constraints:
    #
    #     A_ext * [ x  u  v ] <= b_ext  <=>  {    A * x <= b
    #                                        { |u|, |v| <= box_size
    #
    A_ext = zeros((A.shape[0] + 4, A.shape[1] + 2))
    A_ext[:-4, :-2] = A
    A_ext[-4, -2] = 1
    A_ext[-3, -2] = -1
    A_ext[-2, -1] = 1
    A_ext[-1, -1] = -1
    A_ext = cvxopt.matrix(A_ext)

    b_ext = zeros(b.shape[0] + 4)
    b_ext[:-4] = b
    b_ext[-4:] = array([box_size] * 4)
    b_ext = cvxopt.matrix(b_ext)

    # Equality constraints:
    #
    #     C_ext * [ x  u  v ] == d_ext  <=>  {    C * x == d
    #                                        { [ u  v ] == E * x + f
    #
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
    return vertices, []


def project_polytope_cdd(A, b, C, d, E, f):
    """
    Project a polytope using cdd.

    The initial polytope is defined by:

        A * x <= b
        C * x == d  (optional, disabled by setting C or d to None)

    The output (projection) is computed from:

        y = E * x + f

    See <http://www.roboticsproceedings.org/rss11/p28.pdf> for details.
    """
    b = b.reshape((b.shape[0], 1))

    # the input [b, -A] to cdd.Matrix represents (b - A * x >= 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    linsys = cdd.Matrix(hstack([b, -A]), number_type='float')
    linsys.rep_type = cdd.RepType.INEQUALITY

    if C and d:
        # the input [d, -C] to cdd.Matrix.extend represents (d - C * x == 0)
        # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
        d = d.reshape((d.shape[0], 1))
        linsys.extend(hstack([d, -C]), linear=True)

    # Convert from H- to V-representation
    linsys.canonicalize()
    P = cdd.Polyhedron(linsys)
    generators = P.get_generators()
    # if generators.lin_set:
    #    print "Generators have linear set:", generators.lin_set
    V = array(generators)

    # Project output wrenches to 2D set
    vertices, rays = [], []
    free_coordinates = []
    for i in xrange(V.shape[0]):
        # f_gi = dot(Gs, V[i, 1:])[:3]
        if generators.lin_set and i in generators.lin_set:
            free_coordinates.append(list(V[i, 1:]).index(1.))
        elif V[i, 0] == 1:  # vertex
            # p_Z = (z_Z - z_G) * f_gi / (- mass * 9.81) + p_G
            p_Z = dot(E, V[i, 1:]) + f
            vertices.append(p_Z)
        else:  # ray
            # r_Z = (z_Z - z_G) * f_gi / (- mass * 9.81)
            r_Z = dot(E, V[i, 1:])
            rays.append(r_Z)
    return vertices, rays
