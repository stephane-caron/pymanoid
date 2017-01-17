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

from numpy import array, dot, hstack, sqrt
from pyclipper import Pyclipper, PT_CLIP, PT_SUBJECT, CT_INTERSECTION
from pyclipper import scale_to_clipper, scale_from_clipper
from scipy.spatial import ConvexHull
from shapely.geometry import LineString as ShapelyLineString
from shapely.geometry import Polygon as ShapelyPolygon

from polytope import Polytope


def norm(v):
    return sqrt(dot(v, v))


def __compute_polygon_hull(B, c):
    """
    Compute the vertex representation of a polygon defined by:

        B * x <= c

    where x is a 2D vector.

    NB: the origin [0, 0] should lie inside the polygon (c >= 0) in order to
    build the polar form. If you don't have this guarantee, call
    compute_polar_polygon() instead.

    INPUT:

    - ``B`` -- (2 x K) matrix
    - ``c`` -- vector of length K and positive coordinates

    OUTPUT:

    List of 2D vertices in counterclowise order.

    .. NOTE::

        Checking that (c > 0) is not optional. The rest of the algorithm can be
        executed when some coordinates c_i < 0, but the result would be wrong.
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

    where :math:`x` is a 2D vector.

    Parameters
    ----------

    B : ndarray
        (2 x K) linear inequality matrix
    c : ndarray
        vector of length K and positive coordinates

    Returns
    -------
    vertices : sequence of 2D ndarrays
        List of 2D vertices in counterclowise order.
    """
    x = None
    if not all(c > 0):
        x = Polytope.compute_chebyshev_center(B, c)
        c = c - dot(B, x)
    vertices = __compute_polygon_hull(B, c)
    if x is not None:
        vertices = [v + x for v in vertices]
    return vertices


def intersect_line_polygon_shapely(line, vertices):
    """
    Intersect a line segment with a polygon.

    INPUT:

    - ``line`` -- list of two points
    - ``vertices`` -- vertices of the polygon

    OUTPUT:

    Returns a numpy array of shape (2,).
    """
    def in_line(p):
        for q in line:
            if abs(p[0] - q[0]) < 1e-5 and abs(p[1] - q[1]) < 1e-5:
                return True
        return False

    s_polygon = ShapelyPolygon(vertices)
    s_line = ShapelyLineString(line)
    try:
        coords = (array(p) for p in s_polygon.intersection(s_line).coords)
        coords = [p for p in coords if not in_line(p)]
    except NotImplementedError:
        coords = []
    return coords


def intersect_line_polygon(p1, p2, points):
    """
    Intersect the line segment [p1, p2] with a polygon. If the intersection has
    two points, returns the one closest to p1.

    INPUT:

    - ``p1`` -- end point of line segment (2D or 3D)
    - ``p2`` -- end point of line segment (2D or 3D)
    - ``points`` -- vertices of the polygon

    OUTPUT:

    None if the intersection is empty, otherwise its point closest to p1.

    .. NOTE::

        Adapted from <http://stackoverflow.com/a/20679579>. This variant
        %timeits around 90 us on my machine, vs. 170 us when using shapely.
    """
    def line(p1, p2):
        A = (p1[1] - p2[1])
        B = (p2[0] - p1[0])
        C = (p1[0]*p2[1] - p2[0]*p1[1])
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

    def l1_norm(p, q):
        return abs(p[0] - q[0]) + abs(p[1] - q[1])

    hull = ConvexHull(points)
    vertices = [points[i] for i in hull.vertices]
    n = len(vertices)
    L1 = line(p1, p2)
    x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
    y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
    closest_point = None
    for i, v1 in enumerate(vertices):
        v2 = vertices[(i + 1) % n]
        L2 = line(v1, v2)
        p = intersection(L1, L2)
        if p is not None:
            if not (x_min <= p[0] <= x_max and y_min <= p[1] <= y_max):
                continue
            vx_min, vx_max = min(v1[0], v2[0]), max(v1[0], v2[0])
            vy_min, vy_max = min(v1[1], v2[1]), max(v1[1], v2[1])
            if not (vx_min <= p[0] <= vx_max and vy_min <= p[1] <= vy_max):
                continue
            if closest_point is None \
                    or l1_norm(p, p1) < l1_norm(closest_point, p1):
                closest_point = p
    return array(closest_point) if closest_point else None


def intersect_line_cylinder(p1, p2, points):
    """
    Intersect the line segment [p1, p2] with a vertical cylinder of polygonal
    cross-section. If the intersection has two points, returns the one closest
    to p1.

    INPUT:

    - ``p1`` -- 3D end point of line segment
    - ``p2`` -- 3D end point of line segment
    - ``points`` -- 2D vertices of the polygon

    OUTPUT:

    None if the intersection is empty, otherwise its point closest to p1.
    """
    p = intersect_line_polygon(p1, p2, points)
    if p is None:
        return None
    p1 = array(p1)
    p2 = array(p2)
    alpha = norm(p - p1[:2]) / norm(p2[:2] - p1[:2])
    z = p1[2] + alpha * (p2[2] - p1[2])
    return array([p[0], p[1], z])


def intersect_polygons(polygon1, polygon2):
    """
    Intersect two polygons.

    INPUT:

    - ``polygon1`` -- list of vertices in counterclockwise order
    - ``polygon2`` -- same

    OUTPUT:

    Vertices of the intersection in counterclockwise order.
    """
    # could be accelerated by removing the scale_to/from_clipper()
    subj, clip = (polygon1,), polygon2
    pc = Pyclipper()
    pc.AddPath(scale_to_clipper(clip), PT_CLIP)
    pc.AddPaths(scale_to_clipper(subj), PT_SUBJECT)
    solution = pc.Execute(CT_INTERSECTION)
    if not solution:
        return []
    return scale_from_clipper(solution)[0]
