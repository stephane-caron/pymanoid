#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
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

import cdd
import os
import cvxopt
import sys

from numpy import array, dot, hstack, vstack, zeros

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/..')

from polyhedra import compute_bretl_projection
from polyhedra.polygon import compute_polar_polygon


BB_SIZE = 50  # [m], bounding box size


def compute_sep_bretl(contact_set):
    """
    Compute the static-equilibrium polygon of the COM using Bretl & Lall's
    projection method.

    INPUT:

    - ``contact_set`` -- a ContactSet instance

    OUTPUT:

    List of vertices of the static-equilibrium polygon.

    ALGORITHM:

    This projection method is described in [BL08].

    REFERENCES:

    .. [BL08] https://dx.doi.org/10.1109/TRO.2008.2001360
    """
    mass = 42.  # [kg]
    # mass has no effect on the output polygon, see Section IV.B in
    # <https://hal.archives-ouvertes.fr/hal-01349880> for details

    G = contact_set.compute_grasp_matrix(zeros(3))
    F = -contact_set.compute_stacked_wrench_faces()

    # Inequality constraints on [f_all, u, v]
    lp_G = zeros((F.shape[0]+4, F.shape[1]+2))
    lp_G[:-4, :-2] = F
    lp_G[-4, -2] = 1
    lp_G[-3, -2] = -1
    lp_G[-2, -1] = 1
    lp_G[-1, -1] = -1
    lp_G = cvxopt.matrix(lp_G)
    lp_h = zeros(F.shape[0]+4)
    lp_h[-4:] = array([BB_SIZE, BB_SIZE, BB_SIZE, BB_SIZE])
    lp_h = cvxopt.matrix(lp_h)

    # Equality constraints on [f_all, u, v]
    C = G[(0, 1, 2, 5), :]
    D = 1. / (-mass * 9.81) * vstack([-G[4, :], +G[3, :]])
    lp_A = zeros((C.shape[0]+2, C.shape[1]+2))
    lp_A[:-2, :-2] = C
    lp_A[-2:, :-2] = D
    lp_A[-2:, -2:] = array([[-1, 0], [0, -1]])
    lp_A = cvxopt.matrix(lp_A)
    d = -array([0, 0, mass * 9.81, 0])
    lp_b = zeros(C.shape[0]+2)
    lp_b[:-2] = d
    lp_b = cvxopt.matrix(lp_b)

    lp_q = cvxopt.matrix(zeros(F.shape[1]+2))

    lp = lp_q, lp_G, lp_h, lp_A, lp_b

    P = compute_bretl_projection(lp)
    P.sort_vertices()
    vertices_list = P.export_vertices()
    vertices = [array([v.x, v.y]) for v in vertices_list]
    return vertices


def compute_sep_cdd(contact_set):
    """
    Compute the static-equilibrium polygon of the COM using the
    double-description method.

    INPUT:

    - ``contact_set`` -- a ContactSet instance

    OUTPUT:

    List of vertices of the static-equilibrium polygon.

    ALGORITHM:

    Uses the double-description method as described in [CPN16].

    REFERENCES:

    .. [CPN16] https://dx.doi.org/10.1109/TRO.2016.2623338
    """
    mass = 42.  # [kg]
    # mass has no effect on the output polygon, see Section IV.B in
    # <https://hal.archives-ouvertes.fr/hal-01349880> for details

    G = contact_set.compute_grasp_matrix([0, 0, 0])
    A = contact_set.compute_stacked_wrench_faces()
    b = zeros((A.shape[0], 1))
    # the input [b, -A] to cdd.Matrix represents (b - A x >= 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    M = cdd.Matrix(hstack([b, -A]), number_type='float')
    M.rep_type = cdd.RepType.INEQUALITY

    # Equalities:  C [GAW_1 GAW_2 ...] + d == 0
    C = G[(0, 1, 2, 5), :]
    d = array([0, 0, mass * 9.81, 0]).reshape((4, 1))
    # the input [d, -C] to cdd.Matrix.extend represents (d - C x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    M.extend(hstack([d, -C]), linear=True)

    # Convert from H- to V-representation
    P = cdd.Polyhedron(M)
    V = array(P.get_generators())
    if V.shape[0] < 1:
        return [], []

    # COM position from GAW:  [pGx, pGy] = D * [GAW_1 GAW_2 ...]
    D = 1. / (mass * 9.81) * vstack([-G[4, :], +G[3, :]])
    vertices = []
    for i in xrange(V.shape[0]):
        # assert V[i, 0] == 1, "There should be no ray in this polygon"
        p = dot(D, V[i, 1:])
        vertices.append([p[0], p[1]])
    return vertices


def compute_sep_hull(contact_set):
    """
    Compute the static-equilibrium polygon of the COM using a convex-hull
    reduction method.

    INPUT:

    - ``contact_set`` -- a ContactSet instance

    OUTPUT:

    List of vertices of the static-equilibrium polygon.

    ALGORITHM:

    This projection method is described in [CK16].

    REFERENCES:

    .. [CK16] https://hal.archives-ouvertes.fr/hal-01349880
    """
    A_O = contact_set.compute_wrench_face([0, 0, 0])
    k, a_Oz, a_x, a_y = A_O.shape[0], A_O[:, 2], A_O[:, 3], A_O[:, 4]
    B, c = hstack([-a_y.reshape((k, 1)), +a_x.reshape((k, 1))]), -a_Oz
    return compute_polar_polygon(B, c)
