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

import bretl
import cdd
import cvxopt

from numpy import array, dot, hstack, zeros


def project_polyhedron(ineq, eq, proj):
    """
    Apply the affine projection :math:`y = E x + f` to the polyhedron defined
    by:

    .. math::

        \\begin{eqnarray}
        A x & \\leq & b \\
        C x & = & d
        \\end{eqnarray}

    Parameters
    ----------
    ineq : pair of arrays
        Pair (`A`, `b`) describing the inequality constraint.
    eq : pair of arrays
        Pair (`C`, `d`) describing the equality constraint.
    proj : pair of arrays
        Pair (`E`, `f`) describing the affine projection.

    Returns
    -------
    vertices : list of arrays
        List of vertices of the projection.
    rays : list of arrays
        List of rays of the projection.
    """
    (A, b), (C, d), (E, f) = ineq, eq, proj
    b = b.reshape((b.shape[0], 1))
    d = d.reshape((d.shape[0], 1))

    # the input [b, -A] to cdd.Matrix represents (b - A * x >= 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    linsys = cdd.Matrix(hstack([b, -A]), number_type='float')
    linsys.rep_type = cdd.RepType.INEQUALITY

    # the input [d, -C] to cdd.Matrix.extend represents (d - C * x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
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


def project_polytope(ineq, eq, proj, method='cdd'):
    """
    Apply the affine projection :math:`y = E x + f` to the polytope defined by:

    .. math::

        \\begin{eqnarray}
        A x & \\leq & b \\
        C x & = & d
        \\end{eqnarray}

    Parameters
    ----------
    ineq : pair of arrays
        Pair (`A`, `b`) describing the inequality constraint.
    eq : pair of arrays
        Pair (`C`, `d`) describing the equality constraint.
    proj : pair of arrays
        Pair (`E`, `f`) describing the affine projection.
    method : string, optional
        Choice between 'bretl' and 'cdd'.

    Returns
    -------
    vertices : list of arrays
        List of vertices of the projection.
    """
    if method == 'bretl':
        return project_polytope_bretl(ineq, eq, proj)
    vertices, rays = project_polyhedron(ineq, eq, proj)
    assert not rays, "Projection is not a polytope"
    return vertices


def project_polytope_bretl(ineq, eq, proj, solver='glpk', max_radius=42.):
    """
    Project a polytope into a 2D polygon using the incremental projection
    algorithm from [BL08]_. The 2D affine projection :math:`y = E x + f` is
    applied to the polyhedron defined by:

    .. math::

        \\begin{eqnarray}
        A x & \\leq & b \\
        C x & = & d
        \\end{eqnarray}

    Parameters
    ----------
    ineq : pair of arrays
        Pair (`A`, `b`) describing the inequality constraint.
    eq : pair of arrays
        Pair (`C`, `d`) describing the equality constraint.
    proj : pair of arrays
        Pair (`E`, `f`) describing the affine projection.
    solver : string, optional
        LP solver to use (default is GLPK).
    max_radius : scalar
        Maximum distance from origin (in [m]) used to make sure the output
        is bounded. Default is 42 [m].

    Returns
    -------
    vertices : list of arrays
        List of vertices of the projected polygon.
    """
    (A, b), (C, d), (E, f) = ineq, eq, proj
    assert E.shape[0] == f.shape[0] == 2

    # Inequality constraints: A_ext * [ x  u  v ] <= b_ext iff
    # (1) A * x <= b and (2) |u|, |v| <= max_radius
    A_ext = zeros((A.shape[0] + 4, A.shape[1] + 2))
    A_ext[:-4, :-2] = A
    A_ext[-4, -2] = 1
    A_ext[-3, -2] = -1
    A_ext[-2, -1] = 1
    A_ext[-1, -1] = -1
    A_ext = cvxopt.matrix(A_ext)

    b_ext = zeros(b.shape[0] + 4)
    b_ext[:-4] = b
    b_ext[-4:] = array([max_radius] * 4)
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
