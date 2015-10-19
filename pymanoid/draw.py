#!/usr/bin/env python

"""
Extra drawing primitives.
"""

import itertools

from numpy import array, int64, vstack, cross, dot
from polytope import convert_cone_to_vertices
from scipy.spatial import ConvexHull
from pymanoid.toolbox import norm


def draw_polyhedron(env, points, color, plot_type, precomp_hull=None,
                    linewidth=1., pointsize=0.02):
    """
    Draw a polyhedron defined as the convex hull of a set of points.

    env -- openravepy environment
    points -- list of 3D points
    color -- RGBA vector
    plot_type -- bitmask with 1 for edges, 2 for surfaces and 4 for summits
    precomp_hull -- used in the 2D case where the hull has zero volume
    linewidth -- openravepy format
    pointsize -- openravepy format

    """
    is_2d = precomp_hull is not None
    hull = precomp_hull if precomp_hull is not None else ConvexHull(points)
    vertices = array([points[i] for i in hull.vertices])
    points = array(points)
    color = array(color)
    handles = []
    if plot_type & 1:  # include edges
        edge_color = color * 0.7
        edge_color[3] = 1.
        edges = vstack([[points[i], points[j]]
                        for s in hull.simplices
                        for (i, j) in itertools.combinations(s, 2)])
        edges = array(edges)
        handles.append(env.drawlinelist(edges, linewidth=linewidth,
                                        colors=edge_color))
    if plot_type & 2:  # include surface
        if is_2d:
            nv = len(vertices)
            indices = array([(0, i, i + 1) for i in xrange(nv - 1)], int64)
            handles.append(env.drawtrimesh(vertices, indices, colors=color))
        else:
            indices = array(hull.simplices, int64)
            handles.append(env.drawtrimesh(points, indices, colors=color))
    if plot_type & 4:  # vertices
        color[3] = 1.
        handles.append(env.plot3(vertices, pointsize=pointsize, drawstyle=1,
                                 colors=color))
    return handles


def draw_polygon(env, points, n, color, plot_type, linewidth=1.,
                 pointsize=0.02):
    """
    Draw a polygon defined as the convex hull of a set of points. The normal
    vector n of the plane containing the polygon must also be supplied.

    env -- openravepy environment
    points -- list of 3D points
    n -- plane normal vector
    color -- RGBA vector
    plot_type -- bitmask with 1 for edges, 2 for surfaces and 4 for summits
    linewidth -- openravepy format
    pointsize -- openravepy format

    """
    t1 = array([n[2] - n[1], n[0] - n[2], n[1] - n[0]])
    t1 /= norm(t1)
    t2 = cross(n, t1)
    points2d = [[dot(t1, x), dot(t2, x)] for x in points]
    hull = ConvexHull(points2d)
    return draw_polyhedron(env, points, color, plot_type, hull, linewidth,
                           pointsize)


def draw_cone(env, vertices, rays, n, color, plot_type):
    """
    Draw a 2D cone defined from its rays and vertices. The normal vector n of
    the plane containing the cone must also be supplied.

    env -- openravepy environment
    vertices -- list of 3D points
    rays -- list of 3D vectors
    n -- plane normal vector
    color -- RGBA vector
    plot_type -- bitmask with 1 for edges, 2 for surfaces and 4 for summits

    """
    if not rays:
        return draw_polygon(
            env, vertices, n, color=color, plot_type=plot_type)
    plot_vertices = convert_cone_to_vertices(vertices, rays)
    return draw_polygon(
        env, plot_vertices, n, color=color, plot_type=plot_type)
