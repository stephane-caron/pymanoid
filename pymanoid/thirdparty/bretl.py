#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Quang-Cuong Pham <cuong.pham@normalesup.org>
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

import cvxopt
import cvxopt.solvers

from numpy import array, cos, cross, pi, sin
from numpy.random import random
from scipy.linalg import norm
from warnings import warn

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output

try:
    import cvxopt.glpk
    GLPK_IF_AVAILABLE = 'glpk'
    # GLPK is the fastest LP solver I could find so far:
    # <https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html>
    # ... however, it's verbose by default, so tell it to STFU:
    cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
    cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # cvxopt 1.1.7
    cvxopt.solvers.options['LPX_K_MSGLEV'] = 0  # previous versions
except ImportError:
    GLPK_IF_AVAILABLE = None


def norm1(v):
    return abs(v[0]) + abs(v[1])


class Vertex:

    def __init__(self, p):
        self.x = p[0]
        self.y = p[1]
        self.next = None
        self.expanded = False

    def length(self):
        return norm([self.x-self.next.x, self.y-self.next.y])

    def expand(self, lp):
        v1 = self
        v2 = self.next
        v = array([v2.y - v1.y, v1.x - v2.x])  # orthogonal direction to edge
        v = 1 / norm(v) * v
        res, z = optimize_direction(v, lp)
        if not res:
            self.expanded = True
            return False, None
        xopt, yopt = z
        if abs(cross([xopt-v1.x, yopt-v1.y], [v1.x-v2.x, v1.y-v2.y])) < 1e-2:
            self.expanded = True
            return False, None
        else:
            vnew = Vertex([xopt, yopt])
            vnew.next = self.next
            self.next = vnew
            self.expanded = False
            return True, vnew


class Polygon:

    def from_vertices(self, v1, v2, v3):
        v1.next = v2
        v2.next = v3
        v3.next = v1
        self.vertices = [v1, v2, v3]

    def all_expanded(self):
        for v in self.vertices:
            if not v.expanded:
                return False
        return True

    def iter_expand(self, qpconstraints, max_iter):
        """
        Returns true if there's a edge that can be expanded, and expands that
        edge, otherwise returns False.
        """
        nb_iter = 0
        v = self.vertices[0]
        while not self.all_expanded() and nb_iter < max_iter:
            if v.expanded:
                v = v.next
                continue
            res, vnew = v.expand(qpconstraints)
            if not res:
                continue
            self.vertices.append(vnew)
            nb_iter += 1

    def sort_vertices(self):
        """
        Export vertices starting from the left-most and going clockwise.
        Assumes all vertices are on the positive halfplane.
        """
        minsd = 1e10
        ibottom = 0
        for i in range(len(self.vertices)):
            v = self.vertices[i]
            if (v.y + v.next.y) < minsd:
                ibottom = i
                minsd = v.y + v.next.y
        for v in self.vertices:
            v.checked = False
        vcur = self.vertices[ibottom]
        newvertices = []
        while not vcur.checked:
            vcur.checked = True
            newvertices.append(vcur)
            vcur = vcur.next
        newvertices.reverse()
        vfirst = newvertices.pop(-1)
        newvertices.insert(0, vfirst)
        self.vertices = newvertices

    def export_vertices(self, threshold=1e-2):
        export_list = [self.vertices[0]]
        for i in range(1, len(self.vertices)-1):
            vcur = self.vertices[i]
            vlast = export_list[-1]
            if norm([vcur.x-vlast.x, vcur.y-vlast.y]) > threshold:
                export_list.append(vcur)
        export_list.append(self.vertices[-1])  # always add last vertex
        return export_list


def optimize_direction(vdir, lp, solver=GLPK_IF_AVAILABLE):
    """
    Optimize in one direction.

    Parameters
    ----------
    vdir : (3,) array
        Direction in which the optimization is performed.
    lp : array tuple
        Tuple `(q, G, h, A, b)` defining the LP. See
        :func:`pymanoid.thirdparty.cvxopt_.solve_lp` for details.
    solver : string, optional
        Backend LP solver to call.

    Returns
    -------
    succ : bool
        Success boolean.
    z : (3,) array, or 0
        Maximum vertex of the polygon in the direction `vdir`, or 0 in case of
        solver failure.
    """
    lp_q, lp_Gextended, lp_hextended, lp_A, lp_b = lp
    lp_q[-2] = -vdir[0]
    lp_q[-1] = -vdir[1]
    try:
        sol = cvxopt.solvers.lp(
            lp_q, lp_Gextended, lp_hextended, lp_A, lp_b, solver=solver)
        if sol['status'] == 'optimal':
            z = sol['x']
            z = array(z).reshape((lp_q.size[0], ))
            return True, z[-2:]
        else:
            warn("Failed with status %s\n" % sol['status'])
            return False, 0
    except Exception as inst:
        print inst
        return False, 0


def optimize_angle(theta, lp, solver=GLPK_IF_AVAILABLE):
    """
    Optimize in one direction.

    Parameters
    ----------
    theta : scalar
        Angle of the direction in which the optimization is performed.
    lp : array tuple
        Tuple `(q, G, h, A, b)` defining the LP. See
        :func:`pymanoid.thirdparty.cvxopt_.solve_lp` for details.
    solver : string, optional
        Backend LP solver to call.

    Returns
    -------
    succ : bool
        Success boolean.
    z : (3,) array, or 0
        Maximum vertex of the polygon in the direction `vdir`, or 0 in case of
        solver failure.
    """
    d = array([cos(theta), sin(theta)])
    res, z = optimize_direction(d, lp, solver=solver)
    if not res:
        msg = "thirdparty.bretl: "
        msg += "could not optimize in direction %s (theta=%f)" % (str(d), theta)
        raise Exception(msg)
    return z


def compute_polygon(lp, max_iter=1000, solver=GLPK_IF_AVAILABLE):
    """
    Expand a polygon iteratively.

    Parameters
    ----------
    lp : array tuple
        Tuple `(q, G, h, A, b)` defining the linear program. See
        :func:`pymanoid.thirdparty.cvxopt_.solve_lp` for details.
    max_iter : integer, optional
        Maximum number of calls to the LP solver.
    solver : string, optional
        Name of backend LP solver.

    Returns
    -------
    poly : Polygon
        Output polygon.
    """
    two_pi = 2 * pi
    theta = pi * random()
    init_vertices = [optimize_angle(theta, lp, solver)]
    step = two_pi / 3
    while len(init_vertices) < 3 and max_iter >= 0:
        theta += step
        if theta >= two_pi:
            step *= 0.25 + 0.5 * random()
            theta += step - two_pi
        z = optimize_angle(theta, lp, solver)
        if all([norm(z - z0) > 1e-5 for z0 in init_vertices]):
            init_vertices.append(z)
        max_iter -= 1
    v0 = Vertex(init_vertices[0])
    v1 = Vertex(init_vertices[1])
    v2 = Vertex(init_vertices[2])
    polygon = Polygon()
    polygon.from_vertices(v0, v1, v2)
    polygon.iter_expand(lp, max_iter)
    return polygon
