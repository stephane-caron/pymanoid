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

from numpy import array, dot, hstack, ones, vstack, zeros

from misc import norm
from optim import solve_lp


def compute_chebyshev_center(A, b):
    """
    Compute the Chebyshev center of a polyhedron, that is, the point furthest
    away from all inequalities.

    Parameters
    ----------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.

    Returns
    -------
    z : array, shape=(k,)
        Point further away from all inequalities.

    References
    ----------
    .. [BV04] Stephen Boyd and Lieven Vandenberghe, "Convex Optimization",
              Section 4.3.1, p. 148.
    """
    cost = zeros(A.shape[1] + 1)
    cost[-1] = -1.
    a_cheby = array([norm(A[i, :]) for i in xrange(A.shape[0])])
    A_cheby = hstack([A, a_cheby.reshape((A.shape[0], 1))])
    z = solve_lp(cost, A_cheby, b)
    if z[-1] < -1e-1:  # last coordinate is distance to boundaries
        raise Exception("Polytope is empty (margin violation %.2f)" % z[-1])
    return z[:-1]


def compute_cone_face_matrix(S):
    """
    Compute the face matrix of a polyhedral convex cone from its span matrix.

    Parameters
    ----------
    S : array, shape=(n, m)
        Span matrix defining the cone as :math:`x = S \\lambda` with
        :math:`\\lambda \\geq 0`.

    Returns
    -------
    F : array, shape=(k, n)
        Face matrix defining the cone equivalently by :math:`F x \\leq 0`.
    """
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


def compute_polytope_hrep(vertices):
    """
    Compute the halfspace representation of a polytope defined as
    convex hull of a set of vertices:

    .. math::

        A x \\leq b
        \\quad \\Leftrightarrow \\quad
        x \\in \\mathrm{conv}(\\mathrm{vertices})

    Parameters
    ----------
    vertices : list of arrays
        List of polytope vertices.

    Returns
    -------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.
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


def compute_polytope_vertices(A, b):
    """
    Compute the vertices of a polytope given in halfspace representation by
    :math:`A x \\leq b`.

    Parameters
    ----------
    A : array, shape=(m, k)
        Matrix of halfspace representation.
    b : array, shape=(m,)
        Vector of halfspace representation.

    Returns
    -------
    vertices : list of arrays
        List of polytope vertices.
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
