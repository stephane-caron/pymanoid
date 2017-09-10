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

from __future__ import division

import cdd
import cvxopt

from numpy import array, dot, hstack, ones, vstack, zeros
from scipy.spatial import ConvexHull

from misc import norm
from optim import solve_lp
from thirdparty import bretl


PREC_TOL = 1e-10  # tolerance to numerical imprecisions


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

    Notes
    -----
    The Chebyshev center is discussed in [Boyd04]_, Section 4.3.1, p. 148.
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


def __compute_polygon_hull(B, c):
    """
    Compute the vertex representation of a polygon defined by:

    .. math::

        B x \\leq c

    where `x` is a 2D vector.

    Parameters
    ----------
    B : array, shape=(2, K)
        Linear inequality matrix.
    c : array, shape=(K,)
        Linear inequality vector with positive coordinates.

    Returns
    -------
    vertices : list of arrays
        List of 2D vertices in counterclowise order.

    Notes
    -----
    The origin [0, 0] should lie inside the polygon (:math:`c \\geq 0`) in order
    to build the polar form. If you don't have this guarantee, call
    ``compute_polar_polygon()`` instead.

    Checking that :math:`c > 0` is not optional. The rest of the algorithm can
    be executed when some coordinates :math:`c_i < 0`, but the result would be
    wrong.
    """
    assert B.shape[1] == 2, \
        "Input (B, c) is not a polygon: B.shape = %s" % str(B.shape)
    assert all(c > 0), \
        "Polygon should contain the origin, but min(c) = %.2f" % min(c)

    B_polar = hstack([
        (B[:, column] * 1. / c).reshape((B.shape[0], 1))
        for column in xrange(2)])

    def axis_intersection(i, j):
        ai, bi = c[i], B[i]
        aj, bj = c[j], B[j]
        x = (ai * bj[1] - aj * bi[1]) * 1. / (bi[0] * bj[1] - bj[0] * bi[1])
        y = (bi[0] * aj - bj[0] * ai) * 1. / (bi[0] * bj[1] - bj[0] * bi[1])
        return array([x, y])

    # QHULL OPTIONS:
    #
    # - ``Pp`` -- do not report precision problems
    # - ``Q0`` -- no merging with C-0 and Qx
    #
    # ``Q0`` avoids [this bug](https://github.com/scipy/scipy/issues/6484).
    # It slightly diminishes computation times (0.9 -> 0.8 ms on my machine)
    # but raises QhullError at the first sight of precision errors.
    #
    hull = ConvexHull([row for row in B_polar], qhull_options='Pp Q0')
    #
    # contrary to hull.simplices (which was not in practice), hull.vertices is
    # guaranteed to be in counterclockwise order for 2-D (see scipy doc)
    #
    simplices = [(hull.vertices[i], hull.vertices[i + 1])
                 for i in xrange(len(hull.vertices) - 1)]
    simplices.append((hull.vertices[-1], hull.vertices[0]))
    vertices = [axis_intersection(i, j) for (i, j) in simplices]
    return vertices


def compute_polygon_hull(B, c):
    """
    Compute the vertex representation of a polygon defined by:

    .. math::

        B x \leq c

    where `x` is a 2D vector.

    Parameters
    ----------
    B : array, shape=(2, K)
        Linear inequality matrix.
    c : array, shape=(K,)
        Linear inequality vector.

    Returns
    -------
    vertices : list of arrays
        List of 2D vertices in counterclockwise order.
    """
    x = None
    if not all(c > 0):
        x = compute_chebyshev_center(B, c)
        c = c - dot(B, x)
    if not all(c > 0):
        raise Exception("Polygon is empty (min. dist. to edge %.2f)" % min(c))
    vertices = __compute_polygon_hull(B, c)
    if x is not None:
        vertices = [v + x for v in vertices]
    return vertices


def compute_polytope_hrep(vertices):
    """
    Compute the halfspace representation (H-rep) of a polytope defined as convex
    hull of a set of vertices:

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


def intersect_line_polygon(line, vertices, apply_hull):
    """
    Intersect a line segment with a polygon.

    Parameters
    ----------
    line : couple of arrays
        End points of the line segment (2D or 3D).
    vertices : list of arrays
        Vertices of the polygon.
    apply_hull : bool
        Set to `True` to apply a convex hull algorithm to `vertices`. Otherwise,
        the function assumes that vertices are already sorted in clockwise or
        counterclockwise order.

    Returns
    -------
    inter_points : list of array
        List of intersection points between the line segment and the polygon.

    Notes
    -----
    This code is adapted from <http://stackoverflow.com/a/20679579>. With
    `apply_hull=True`, this variant %timeits around 90 us on my machine, vs. 170
    us when using the Shapely library <http://toblerity.org/shapely/> (the
    latter variant was removed by commit a8a267b). On the same setting with
    `apply_hull=False`, it %timeits to 6 us.
    """
    def line_coordinates(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0] * p2[1] - p2[0] * p1[1])
        return A, B, -C

    def intersection(L1, L2):
        D = L1[0] * L2[1] - L1[1] * L2[0]
        Dx = L1[2] * L2[1] - L1[1] * L2[2]
        Dy = L1[0] * L2[2] - L1[2] * L2[0]
        if abs(D) < 1e-5:
            return None
        x = Dx / D
        y = Dy / D
        return x, y

    if apply_hull:
        points = vertices
        hull = ConvexHull(points)
        vertices = [points[i] for i in hull.vertices]

    n = len(vertices)
    p1, p2 = line
    L1 = line_coordinates(p1, p2)
    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    inter_points = []
    for i, v1 in enumerate(vertices):
        v2 = vertices[(i + 1) % n]
        L2 = line_coordinates(v1, v2)
        p = intersection(L1, L2)
        if p is not None:
            if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max):
                continue
            vx_min, vx_max = min(v1[0], v2[0]), max(v1[0], v2[0])
            vy_min, vy_max = min(v1[1], v2[1]), max(v1[1], v2[1])
            if not (vx_min - PREC_TOL <= p[0] <= vx_max + PREC_TOL and
                    vy_min - PREC_TOL <= p[1] <= vy_max + PREC_TOL):
                continue
            inter_points.append(array(p))
    return inter_points


def intersect_line_cylinder(line, vertices):
    """
    Intersect the line segment [p1, p2] with a vertical cylinder of polygonal
    cross-section. If the intersection has two points, returns the one closest
    to p1.

    Parameters
    ----------
    line : couple of (3,) arrays
        End points of the 3D line segment.
    vertices : list of (3,) arrays
        Vertices of the polygon.

    Returns
    -------
    inter_points : list of (3,) arrays
        List of intersection points between the line segment and the cylinder.
    """
    inter_points = []
    inter_2d = intersect_line_polygon(line, vertices, apply_hull=True)
    for p in inter_2d:
        p1, p2 = array(line[0]), array(line[1])
        alpha = norm(p - p1[:2]) / norm(p2[:2] - p1[:2])
        z = p1[2] + alpha * (p2[2] - p1[2])
        inter_points.append(array([p[0], p[1], z]))
    return inter_points


def intersect_polygons(polygon1, polygon2):
    """
    Intersect two polygons.

    Parameters
    ----------
    polygon1 : list of arrays
        Vertices of the first polygon in counterclockwise order.
    polygon1 : list of arrays
        Vertices of the second polygon in counterclockwise order.

    Returns
    -------
    intersection : list of arrays
        Vertices of the intersection in counterclockwise order.
    """
    from pyclipper import Pyclipper, PT_CLIP, PT_SUBJECT, CT_INTERSECTION
    from pyclipper import scale_to_clipper, scale_from_clipper
    # could be accelerated by removing the scale_to/from_clipper()
    subj, clip = (polygon1,), polygon2
    pc = Pyclipper()
    pc.AddPath(scale_to_clipper(clip), PT_CLIP)
    pc.AddPaths(scale_to_clipper(subj), PT_SUBJECT)
    solution = pc.Execute(CT_INTERSECTION)
    if not solution:
        return []
    return scale_from_clipper(solution)[0]


def project_polyhedron(proj, ineq, eq=None):
    """
    Apply the affine projection :math:`y = E x + f` to the polyhedron defined
    by:

    .. math::

        \\begin{eqnarray}
        A x & \\leq & b \\\\
        C x & = & d
        \\end{eqnarray}

    Parameters
    ----------
    proj : pair of arrays
        Pair (`E`, `f`) describing the affine projection.
    ineq : pair of arrays
        Pair (`A`, `b`) describing the inequality constraint.
    eq : pair of arrays, optional
        Pair (`C`, `d`) describing the equality constraint.

    Returns
    -------
    vertices : list of arrays
        List of vertices of the projection.
    rays : list of arrays
        List of rays of the projection.
    """
    # the input [b, -A] to cdd.Matrix represents (b - A * x >= 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    (A, b) = ineq
    b = b.reshape((b.shape[0], 1))
    linsys = cdd.Matrix(hstack([b, -A]), number_type='float')
    linsys.rep_type = cdd.RepType.INEQUALITY

    # the input [d, -C] to cdd.Matrix.extend represents (d - C * x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    if eq is not None:
        (C, d) = eq
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
    (E, f) = proj
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


def project_polytope(proj, ineq, eq=None, method='cdd'):
    """
    Apply the affine projection :math:`y = E x + f` to the polytope defined by:

    .. math::

        \\begin{eqnarray}
        A x & \\leq & b \\\\
        C x & = & d
        \\end{eqnarray}

    Parameters
    ----------
    proj : pair of arrays
        Pair (`E`, `f`) describing the affine projection.
    ineq : pair of arrays
        Pair (`A`, `b`) describing the inequality constraint.
    eq : pair of arrays, optional
        Pair (`C`, `d`) describing the equality constraint.
    method : string, optional
        Choice between 'bretl' and 'cdd'.

    Returns
    -------
    vertices : list of arrays
        List of vertices of the projection.

    Notes
    -----
    The number of columns of all matrices `A`, `C` and `E` corresponds to the
    dimension of the input space, while the number of lines of `E` corresponds
    to the dimension of the output space.
    """
    if method == 'bretl':
        assert eq is not None, "Bretl method requires = constraints for now"
        return project_polytope_bretl(proj, ineq, eq)
    vertices, rays = project_polyhedron(ineq, eq, proj)
    assert not rays, "Projection is not a polytope"
    return vertices


def project_polytope_bretl(proj, ineq, eq, max_radius=42.):
    """
    Project a polytope into a 2D polygon using the incremental projection
    algorithm from [Bretl08]_. The 2D affine projection :math:`y = E x + f` is
    applied to the polyhedron defined by:

    .. math::

        \\begin{eqnarray}
        A x & \\leq & b \\\\
        C x & = & d
        \\end{eqnarray}

    Parameters
    ----------
    proj : pair of arrays
        Pair (`E`, `f`) describing the affine projection.
    ineq : pair of arrays
        Pair (`A`, `b`) describing the inequality constraint.
    eq : pair of arrays
        Pair (`C`, `d`) describing the equality constraint.
    max_radius : scalar, optional
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
    polygon = bretl.compute_polygon(lp)
    polygon.sort_vertices()
    vertices_list = polygon.export_vertices()
    vertices = [array([v.x, v.y]) for v in vertices_list]
    return vertices
