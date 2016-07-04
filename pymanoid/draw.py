#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
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


import itertools

from env import get_env
from numpy import array, int64, vstack, cross, dot
from scipy.spatial import ConvexHull
from utils import norm

BIG_DIST = 1000.  # [m]


class UnboundedPolyhedron(Exception):

    pass


def _matplotlib_to_rgb(color):
    acolor = [0., 0., 0.]
    if color == 'k':
        return acolor
    if color in ['r', 'm', 'y']:
        acolor[0] += 0.5
    if color in ['g', 'y', 'c']:
        acolor[1] += 0.5
    if color in ['b', 'c', 'm']:
        acolor[2] += 0.5
    return acolor


def _matplotlib_to_rgba(color, alpha=0.5):
    return _matplotlib_to_rgb(color) + [alpha]


def draw_line(p1, p2, color='g', linewidth=1.):
    """
    Draw a line between points p1 and p2.

    INPUT

    - ``p1`` -- one end of the line
    - ``p2`` -- other end of the line
    - ``color`` -- (default: 'g') matplotlib color letter or RGB triplet
    - ``linewidth`` -- thickness of drawn line

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    if type(color) is str:
        color = _matplotlib_to_rgb(color)
    return get_env().drawlinelist(
        array([p1, p2]), linewidth=linewidth, colors=color),


def draw_arrow(p1, p2, color='r', linewidth=0.02):
    """
    Draw an arrow between two points.

    INPUT:

    - ``p1`` -- 3D coordinates of the origin of the arrow
    - ``p2`` -- 3D coordinates of the end of the arrow
    - ``color`` -- (default: 'r') matplotlib color letter or RGB triplet
    - ``linewidth`` -- thickness of force vector

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    if type(color) is str:
        color = _matplotlib_to_rgb(color)
    return get_env().drawarrow(p1, p2, linewidth=linewidth, color=color)


def draw_force(p, f, scale=0.005, color='r', linewidth=0.02):
    """
    Draw a force acting at a given point.

    INPUT:

    - ``p`` -- point where the force is acting
    - ``f`` -- force vector
    - ``color`` -- (default: 'r') matplotlib color letter or RGB triplet
    - ``scale`` -- scaling factor between Euclidean and Force spaces
    - ``linewidth`` -- thickness of force vector

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    if type(color) is str:
        color = _matplotlib_to_rgb(color)
    if dot(f, f) < 1e-10:
        return None
    return get_env().drawarrow(
        p, p + scale * f, linewidth=linewidth, color=color)


def draw_point(p, color='g', pointsize=0.05):
    return draw_points([p], color, pointsize)


def draw_points(points, color='g', pointsize=0.05):
    if type(color) is str:
        color = _matplotlib_to_rgba(color, alpha=1.)
    return get_env().plot3(
        array(points), pointsize=pointsize, drawstyle=1,
        colors=color)


def draw_polyhedron(points, combined='g-#', color=None, faces=None,
                    linewidth=1., pointsize=0.01, hull=None):
    """
    Draw a polyhedron defined as the convex hull of a set of points.

    INPUT:

    - ``points`` -- list of 3D points
    - ``combined`` -- (default: 'g-#') drawing spec in matplotlib fashion: color
      letter followed by faces to draw (see ``color`` and ``faces``)
    - ``color`` -- color letter or RGBA tuple
    - ``faces`` -- string of symbols indicating the faces of the polyhedron to
      draw: use '.' for vertices, '-' for edges and '#' for facets
    - ``hull`` -- used in the 2D case where the hull has zero volume
    - ``linewidth`` -- line thickness in meters
    - ``pointsize`` -- point size in meters

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    is_2d = hull is not None
    if color is None:
        color = combined[0]
    if faces is None:
        faces = combined[1:]
    if type(color) is str:
        color = _matplotlib_to_rgba(color)
    if hull is None:
        hull = ConvexHull(points)
    vertices = array([points[i] for i in hull.vertices])
    points = array(points)
    color = array(color if color is not None else (0.0, 0.5, 0.0, 0.5))
    handles = []
    env = get_env()
    if '-' in faces:  # edges
        edge_color = color * 0.7
        edge_color[3] = 1.
        edges = vstack([[points[i], points[j]]
                        for s in hull.simplices
                        for (i, j) in itertools.combinations(s, 2)])
        edges = array(edges)
        handles.append(env.drawlinelist(
            edges, linewidth=linewidth, colors=edge_color))
    if '#' in faces:  # facets
        if is_2d:
            nv = len(vertices)
            indices = array([(0, i, i + 1) for i in xrange(nv - 1)], int64)
            handles.append(env.drawtrimesh(vertices, indices, colors=color))
        else:
            indices = array(hull.simplices, int64)
            handles.append(env.drawtrimesh(points, indices, colors=color))
    if '.' in faces:  # vertices
        color[:3] *= 0.75
        color[3] = 1.
        handles.append(env.plot3(
            vertices, pointsize=pointsize, drawstyle=1, colors=color))
    return handles


def draw_polygon(points, normal, combined='g-#', color=None, faces=None,
                 linewidth=1., pointsize=0.02):
    """
    Draw a polygon defined as the convex hull of a set of points. The normal
    vector n of the plane containing the polygon must also be supplied.

    INPUT:

    - ``points`` -- list of coplanar 3D points
    - ``normal`` -- unit vector normal to the drawing plane
    - ``combined`` -- (default: 'g-#') drawing spec in matplotlib fashion: color
      letter followed by faces to draw (see ``color`` and ``faces``)
    - ``color`` -- color letter or RGBA tuple
    - ``faces`` -- string of symbols indicating the faces of the polyhedron to
      draw: use '.' for vertices, '-' for edges and '#' for facets
    - ``linewidth`` -- (default: 1.) thickness of drawn line
    - ``pointsize`` -- (default: 0.02) vertex size

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    n = normal
    t1 = array([n[2] - n[1], n[0] - n[2], n[1] - n[0]], dtype=float)
    t1 /= norm(t1)
    t2 = cross(n, t1)
    points2d = [[dot(t1, x), dot(t2, x)] for x in points]
    hull = ConvexHull(points2d)
    return draw_polyhedron(
        points, combined, color, faces, linewidth, pointsize, hull=hull)


def pick_2d_extreme_rays(rays):
    if len(rays) <= 2:
        return rays
    u_high, u_low = None, rays.pop()
    while u_high is None:
        ray = rays.pop()
        c = cross(u_low, ray)
        if abs(c) < 1e-4:
            continue
        elif c < 0:
            u_low, u_high = ray, u_low
        else:
            u_high = ray
    for u in rays:
        c1 = cross(u_low, u)
        c2 = cross(u_high, u)
        if c1 < 0 and c2 < 0:
            u_low = u
        elif c1 > 0 and c2 > 0:
            u_high = u
        elif c1 < 0 and c2 > 0:
            raise UnboundedPolyhedron
    return u_low, u_high


def _convert_cone2d_to_vertices(vertices, rays):
    if not rays:
        return vertices
    try:
        r0, r1 = pick_2d_extreme_rays([r[:2] for r in rays])
    except UnboundedPolyhedron:
        vertices = [
            BIG_DIST * array([-1, -1]),
            BIG_DIST * array([-1, +1]),
            BIG_DIST * array([+1, -1]),
            BIG_DIST * array([+1, +1])]
        return vertices, []
    r0 = array([r0[0], r0[1], 0.])
    r1 = array([r1[0], r1[1], 0.])
    r0 = r0 / norm(r0)
    r1 = r1 / norm(r1)
    conv_vertices = [v for v in vertices]
    conv_vertices += [v + r0 * BIG_DIST for v in vertices]
    conv_vertices += [v + r1 * BIG_DIST for v in vertices]
    return conv_vertices


def draw_2d_cone(vertices, rays, normal, combined='g-#', color=None,
                 faces=None):
    """
    Draw a 2D cone defined from its rays and vertices. The normal vector n of
    the plane containing the cone must also be supplied.

    INPUT:

    - ``vertices`` -- list of 3D points
    - ``rays`` -- list of 3D vectors
    - ``normal`` -- unit vector normal to the drawing plane
    - ``combined`` -- (default: 'g-#') drawing spec in matplotlib fashion: color
      letter followed by faces to draw (see ``color`` and ``faces``)
    - ``color`` -- color letter or RGBA tuple
    - ``faces`` -- string of symbols indicating the faces of the polyhedron to
      draw: use '.' for vertices, '-' for edges and '#' for facets

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    if not rays:
        return draw_polygon(vertices, normal, combined, color, faces)
    plot_vertices = _convert_cone2d_to_vertices(vertices, rays)
    return draw_polygon(plot_vertices, normal, combined, color, faces)
