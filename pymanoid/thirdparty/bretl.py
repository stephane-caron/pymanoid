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
from pylab import double, hold, plot
from scipy.linalg import norm
from warnings import warn
from StringIO import StringIO

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
        v = array([v2.y-v1.y, v1.x-v2.x])  # orthogonal direction to edge
        v = 1 / norm(v) * v
        res, z = OptimizeDirection(v, lp)
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

    def Plot(self):
        plot([self.x, self.next.x], [self.y, self.next.y])

    def Print(self):
        print self.x, self.y, "to", self.next.x, self.next.y


class Polygon:

    def fromVertices(self, v1, v2, v3):
        v1.next = v2
        v2.next = v3
        v3.next = v1
        self.vertices = [v1, v2, v3]

    def fromString(self, s):
        buff = StringIO(s)
        self.vertices = []
        while(True):
            l = buff.readline()
            l = l.strip(" \n")
            if len(l) < 2:
                break
            x, y = [double(x) for x in l.split(' ')]
            vnew = Vertex([x, y])
            self.vertices.append(vnew)

        for i in range(len(self.vertices)-1):
            self.vertices[i].next = self.vertices[i+1]
        self.vertices[-1].next = self.vertices[0]

    def all_expanded(self):
        for v in self.vertices:
            if not v.expanded:
                return False
        return True

    def iter_expand(self, qpconstraints, maxiter=10):
        """
        Returns true if there's a edge that can be expanded, and expands that
        edge, otherwise returns False.
        """
        niter = 0
        v = self.vertices[0]
        while not self.all_expanded() and niter < maxiter:
            if not v.expanded:
                res, vnew = v.expand(qpconstraints)
                if res:
                    self.vertices.append(vnew)
                    niter += 1
            else:
                v = v.next

    def sort_vertices(self):
        """
        Export the vertices starting from the left-most and going clockwise.
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
            if norm([vcur.x-vlast.x, vcur.y-vlast.y]) > threshold:
                export_list.append(vcur)
        export_list.append(self.vertices[-1])  # always add last vertex
        return export_list

    def Plot(self):
        hold("on")
        for v in self.vertices:
            v.Plot()

    def Print(self):
        print "Polygon contains vertices"
        for v in self.vertices:
            v.Print()


def OptimizeDirection(vdir, lp, solver=GLPK_IF_AVAILABLE):
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


def ComputePolygon(lp, solver=GLPK_IF_AVAILABLE):
    """
    Expand a polygon iteratively.

    Parameters
    ----------
    lp : array tuple
        Tuple `(q, G, h, A, b)` defining the LP. See
        :func:`pymanoid.thirdparty.cvxopt_.solve_lp` for details.
    solver : string, optional
        Name of backend LP solver.

    Returns
    -------
    succ : bool
        Success boolean.
    poly : Polygon
        Output polygon.
    """
    theta = pi * random()
    d1 = array([cos(theta), sin(theta)])
    d2 = array([cos(theta + 2 * pi / 3), sin(theta + 2 * pi / 3)])
    d3 = array([cos(theta + 4 * pi / 3), sin(theta + 4 * pi / 3)])
    res, z1 = OptimizeDirection(d1, lp, solver=solver)
    if not res:
        return False, d1
    res, z2 = OptimizeDirection(d2, lp, solver=solver)
    if not res:
        return False, d2
    res, z3 = OptimizeDirection(d3, lp, solver=solver)
    if not res:
        return False, d3
    v1 = Vertex(z1)
    v2 = Vertex(z2)
    v3 = Vertex(z3)
    poly = Polygon()
    poly.fromVertices(v1, v2, v3)
    poly.iter_expand(lp, 1000)
    return True, poly
