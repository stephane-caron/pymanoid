#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Quang-Cuong Pham <cuong.pham@normalesup.org>
#
# This file is part of pymanoid.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE.  See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

import cvxopt
import cvxopt.solvers

from numpy import array, cos, cross, pi, sin
from numpy.random import random
from scipy.linalg import norm
from warnings import warn


# STFU GLPK:
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # cvxopt 1.1.7
cvxopt.solvers.options['LPX_K_MSGLEV'] = 0  # previous versions

# GLPK is the fastest LP solver we could find so far,
# see <https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html>


class ExpansionError(Exception):

    def __init__(self, vdir):
        self.vdir = vdir


def compute_bretl_projection(lp, solver='glpk'):
    """
    Expand a polygon iteratively using Bretl & Lall's algorithm [BL08].

    INPUT:

    - ``lp`` -- tuple (q, G, h, A, b) of LP matrices
    - ``solver`` -- (optional) LP backend for CVXOPT

    REFERENCES:

    .. [BL08]  https://dx.doi.org/10.1109/TRO.2008.2001360
    """
    theta = pi * random()
    d1 = array([cos(theta), sin(theta)])
    d2 = array([cos(theta + 2 * pi / 3), sin(theta + 2 * pi / 3)])
    d3 = array([cos(theta + 4 * pi / 3), sin(theta + 4 * pi / 3)])
    res, z1 = optimize_direction(d1, lp, solver=solver)
    if not res:
        raise ExpansionError(d1)
    res, z2 = optimize_direction(d2, lp, solver=solver)
    if not res:
        raise ExpansionError(d2)
    res, z3 = optimize_direction(d3, lp, solver=solver)
    if not res:
        raise ExpansionError(d3)
    v1 = Vertex(z1)
    v2 = Vertex(z2)
    v3 = Vertex(z3)
    P0 = Polygon()
    P0.fromVertices(v1, v2, v3)
    P0.iter_expand(lp, 1000)
    return P0


def optimize_direction(vdir, lp, solver='glpk'):
    """
    Optimize in one direction.

    INPUT:

    - ``vdir`` -- direction in which to optimize
    - ``lp`` -- tuple (q, G, h, A, b) of LP matrices
    - ``solver`` -- (optional) LP backend for CVXOPT

    OUTPUT:

    A tuple ``(b, z)`` where ``b`` is a success boolean and ``z`` is the optimal
    vector.
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
        warn("Exception: %s" % repr(inst))
        return False, 0


class Vertex:

    def __init__(self, p):
        """
        Create new vertex from iterable.

        INPUT:

        - ``p`` -- iterable, e.g. a list or numpy.ndarray
        """
        self.x = p[0]
        self.y = p[1]
        self.next = None
        self.expanded = False

    def length(self):
        """
        Length of the edge following the vertex in the overall polygon.
        """
        return norm([self.x - self.next.x, self.y - self.next.y])

    def expand(self, lp):
        """
        Expand polygon from a given LP.

        INPUT:

        - ``lp`` -- linear program used for expansion
        """
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


class Polygon(object):

    def fromVertices(self, v1, v2, v3):
        v1.next = v2
        v2.next = v3
        v3.next = v1
        self.vertices = [v1, v2, v3]

    def all_expanded(self):
        for v in self.vertices:
            if not v.expanded:
                return False
        return True

    def iter_expand(self, constraints, maxiter=10):
        """
        Returns ``True`` if there's a edge that can be expanded, and expands
        that edge, otherwise returns ``False``.
        """
        niter = 0
        v = self.vertices[0]
        while not self.all_expanded() and niter < maxiter:
            if not v.expanded:
                res, vnew = v.expand(constraints)
                if res:
                    self.vertices.append(vnew)
                    niter += 1
            else:
                v = v.next

    def sort_vertices(self):
        """
        Export the vertices starting from the left-most and going clockwise.

        .. NOTE::

            Assumes every vertices are on the positive halfplane.
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
            if norm([vcur.x - vlast.x, vcur.y - vlast.y]) > threshold:
                export_list.append(vcur)
        # always add last vertex
        export_list.append(self.vertices[-1])
        return export_list
