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

import itertools

from numpy import array, cross, dot, int64, sqrt, vstack
from scipy.spatial import ConvexHull
from scipy.spatial.qhull import QhullError
from warnings import warn

from misc import norm
from sim import get_openrave_env


BIG_DIST = 1000.  # [m]


class UnboundedPolyhedron(Exception):

    pass


def matplotlib_to_rgb(color):
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


def matplotlib_to_rgba(color, alpha=0.5):
    return matplotlib_to_rgb(color) + [alpha]


def draw_2d_cone(vertices, rays, normal, combined='g-#', color=None,
                 faces=None):
    """
    Draw a 2D cone defined from its rays and vertices. The normal vector n of
    the plane containing the cone must also be supplied.

    INPUT:

    - ``vertices`` -- list of 3D points
    - ``rays`` -- list of 3D vectors
    - ``normal`` -- unit vector normal to the drawing plane
    - ``combined`` -- (default: 'g-#') drawing spec in matplotlib fashion
    - ``color`` -- color letter or RGBA tuple
    - ``faces`` -- string indicating the faces of the polyhedron to draw

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    if not rays:
        return draw_polygon(vertices, normal, combined, color, faces)
    plot_vertices = _convert_cone2d_to_vertices(vertices, rays)
    return draw_polygon(plot_vertices, normal, combined, color, faces)


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
        color = matplotlib_to_rgb(color)
    env = get_openrave_env()
    return env.drawarrow(p1, p2, linewidth=linewidth, color=color)


def draw_force(point, force, scale=0.005, linewidth=0.015):
    """
    Draw a force acting at a given point.

    INPUT:

    - ``point`` -- point where the force is acting
    - ``force`` -- 3D force vector
    - ``scale`` -- force-to-distance scaling factor in [N] / [m]
    - ``linewidth`` -- thickness of force vector

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    f_scale = scale * force
    if dot(f_scale, f_scale) < 1e-6:
        return None
    return draw_arrow(point, point + f_scale, color='r', linewidth=linewidth)


def draw_line(start_point, end_point, color='g', linewidth=1.):
    """
    Draw a line between two points.

    INPUT:

    - ``start_point`` -- one end of the line
    - ``end_point`` -- other end of the line
    - ``color`` -- (default: 'g') matplotlib color letter or RGB triplet
    - ``linewidth`` -- thickness of drawn line

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    if type(color) is str:
        color = matplotlib_to_rgb(color)
    return get_openrave_env().drawlinelist(
        array([start_point, end_point]), linewidth=linewidth, colors=color)


def draw_point(point, color='g', pointsize=0.05):
    """
    Draw a point.

    INPUT:

    - ``point`` -- 3D vector of point coordinates
    - ``pointsize`` -- point radius in [m]

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    return draw_points([point], color, pointsize)


def draw_points(points, color='g', pointsize=0.05):
    """
    Draw a list of points.

    INPUT:

    - ``points`` -- list of 3D vectors of point coordinates
    - ``pointsize`` -- point radius in [m]

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    if type(color) is str:
        color = matplotlib_to_rgba(color, alpha=1.)
    return get_openrave_env().plot3(
        array(points), pointsize=pointsize, drawstyle=1,
        colors=color)


def draw_polygon(points, normal, combined='g-#', color=None, faces=None,
                 linewidth=1., pointsize=0.02):
    """
    Draw a polygon defined as the convex hull of a set of points. The normal
    vector n of the plane containing the polygon must also be supplied.

    INPUT:

    - ``points`` -- list of coplanar 3D points
    - ``normal`` -- unit vector normal to the drawing plane
    - ``combined`` -- (default: 'g-#') drawing spec in matplotlib fashion
    - ``color`` -- color letter or RGBA tuple
    - ``faces`` -- string indicating the faces of the polyhedron to draw
    - ``linewidth`` -- (default: 1.) thickness of drawn line
    - ``pointsize`` -- (default: 0.02) vertex size

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.
    """
    assert abs(1. - norm(normal)) < 1e-10
    n = normal
    t = array([n[2] - n[1], n[0] - n[2], n[1] - n[0]], dtype=float)
    t /= norm(t)
    b = cross(n, t)
    points2d = [[dot(t, x), dot(b, x)] for x in points]
    try:
        hull = ConvexHull(points2d)
    except QhullError:
        warn("QhullError: maybe polygon is empty?")
        return []
    except IndexError:
        warn("Qhull raised an IndexError for points2d=%s" % repr(points2d))
        return []
    return draw_polyhedron(
        points, combined, color, faces, linewidth, pointsize, hull=hull)


def draw_polyhedron(points, combined='g-#', color=None, faces=None,
                    linewidth=1., pointsize=0.01, hull=None):
    """
    Draw a polyhedron defined as the convex hull of a set of points.

    INPUT:

    - ``points`` -- list of 3D points
    - ``combined`` -- (default: 'g-#') drawing spec in matplotlib fashion
    - ``color`` -- color letter or RGBA tuple
    - ``faces`` -- string indicating the faces of the polyhedron to draw
    - ``hull`` -- used in the 2D case where the hull has zero volume
    - ``linewidth`` -- line thickness in meters
    - ``pointsize`` -- point size in meters

    OUTPUT:

    And OpenRAVE handle. Must be stored in some variable, otherwise the drawn
    object will vanish instantly.

    .. NOTE::

        In the ``faces`` or ``combined`` strings, use '.' for vertices, '-' for
        edges and '#' for facets.
    """
    is_2d = hull is not None
    if color is None:
        color = combined[0]
    if faces is None:
        faces = combined[1:]
    if type(color) is str:
        color = matplotlib_to_rgba(color)
    if hull is None:
        try:
            hull = ConvexHull(points)
        except QhullError:
            warn("QhullError: maybe polyhedron is empty?")
            return []
    vertices = array([points[i] for i in hull.vertices])
    points = array(points)
    color = array(color if color is not None else (0.0, 0.5, 0.0, 0.5))
    handles = []
    env = get_openrave_env()
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


def draw_wrench(body, wrench, scale=0.005, pointsize=0.02, linewidth=0.01):
    """
    Draw a wrench acting on a given rigid body.

    Parameters
    ----------
    body : Body
        Body on which the wrench is acting.
    force : ndarray
        6D wrench vector in world-frame coordinates.
    scale : scalar
        Scaling factor between Euclidean and Force spaces.
    pointsize : scalar
        Point radius in [m].
    linewidth : scalar
        Thickness of force vector.

    Returns
    -------
    handles : list of OpenRAVE handles
        This list must be stored in some variable, otherwise the drawn object
        will vanish instantly.
    """
    if type(wrench) is list:
        wrench = array(wrench)
    assert wrench.shape == (6,)
    f, tau = wrench[:3], wrench[3:]
    cop = body.p + cross(body.n, tau) / dot(body.n, f)
    tau_z = dot(body.n, tau)
    h1 = draw_point(cop, pointsize=pointsize)
    h2 = draw_force(cop, f, scale=scale, linewidth=linewidth)
    if abs(tau_z) > 1e-1:
        h3 = draw_arrow(
            cop, cop + 10 * scale * tau_z * body.n, color='b',
            linewidth=linewidth)
        return [h1, h2, h3]
    return [h1, h2]


def _convert_cone2d_to_vertices(vertices, rays):
    if not rays:
        return vertices
    try:
        r0, r1 = _pick_2d_extreme_rays([r[:2] for r in rays])
    except UnboundedPolyhedron:
        vertices = [
            BIG_DIST * array([-1, -1]),
            BIG_DIST * array([-1, +1]),
            BIG_DIST * array([+1, -1]),
            BIG_DIST * array([+1, +1])]
        return vertices, []
    r0 = array([r0[0], r0[1], 0.])
    r1 = array([r1[0], r1[1], 0.])
    r0 = r0 / sqrt(dot(r0, r0))
    r1 = r1 / sqrt(dot(r1, r1))
    conv_vertices = [v for v in vertices]
    conv_vertices += [v + r0 * BIG_DIST for v in vertices]
    conv_vertices += [v + r1 * BIG_DIST for v in vertices]
    return conv_vertices


def _pick_2d_extreme_rays(rays):
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
