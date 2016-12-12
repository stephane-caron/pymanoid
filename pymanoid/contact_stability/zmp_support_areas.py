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
import cvxopt
import os
import sys

from numpy import array, dot, eye, hstack, vstack, zeros

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/..')

from polyhedra import compute_bretl_projection


BB_SIZE = 50  # [m], bounding box size


def compute_zmp_area_bretl(contact_set, com, plane):
    """
    Compute the pendular ZMP support area for a given COM position.

    INPUT:

    - ``contact_set`` -- a ContactSet instance
    - ``com`` -- COM position
    - ``plane`` -- position of horizontal plane

    OUTPUT:

    List of vertices of the area.

    ALGORITHM:

    This method relies on Bretl & Lall's projection method [BL08].

    REFERENCES:

    .. [BL08]  https://dx.doi.org/10.1109/TRO.2008.2001360
    """
    mass = 42.  # [kg]
    # mass has no effect on the output polygon, c.f. Section IV.C in
    # <https://hal.archives-ouvertes.fr/hal-01349880>
    z_in, z_out = com[2], plane[2]
    # n = [0, 0, 1]
    crossmat_n = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    G = contact_set.compute_grasp_matrix([0, 0, 0])
    F = -contact_set.compute_stacked_wrench_faces()

    # Inequality constraints on [f_all, u, v]
    lp_G = zeros((F.shape[0]+4, F.shape[1]+2))
    lp_G[:-4, :-2] = F
    lp_G[-4, -2] = 1
    lp_G[-3, -2] = -1
    lp_G[-2, -1] = 1
    lp_G[-1, -1] = -1
    lp_G = cvxopt.matrix(lp_G)
    lp_h = zeros(F.shape[0] + 4)
    lp_h[-4:] = array([BB_SIZE, BB_SIZE, BB_SIZE, BB_SIZE])
    lp_h = cvxopt.matrix(lp_h)

    # Equality constraints on [f_all, u, v]
    B = vstack([
        hstack([z_in * eye(3), crossmat_n]),
        hstack([zeros(3), com.p])])  # hstack([-(cross(n, p_in)), n])])
    C = 1. / (- mass * 9.81) * dot(B, G)
    D = (z_out - z_in) / (-mass * 9.81) * G[:2, :]
    lp_A = zeros((C.shape[0]+2, C.shape[1]+2))
    lp_A[:-2, :-2] = C
    lp_A[-2:, :-2] = D
    lp_A[-2:, -2:] = array([[-1, 0], [0, -1]])
    lp_A = cvxopt.matrix(lp_A)
    d = hstack([com.p, [0]])
    lp_b = zeros(C.shape[0]+2)
    lp_b[:-2] = d
    lp_b[-2:] = -com.p[:2]
    lp_b = cvxopt.matrix(lp_b)

    lp_q = cvxopt.matrix(zeros(F.shape[1]+2))

    lp = lp_q, lp_G, lp_h, lp_A, lp_b

    P = compute_bretl_projection(lp)
    P.sort_vertices()
    vertices_list = P.export_vertices()
    vertices = [array([v.x, v.y]) for v in vertices_list]
    return vertices


def compute_zmp_area_cdd(contact_set, com, plane):
    """
    Compute the pendular ZMP support area for a given COM position.

    INPUT:

    - ``contact_set`` -- a ContactSet instance
    - ``com`` -- COM position
    - ``plane`` -- position of horizontal plane

    OUTPUT:

    List of vertices of the area.

    ALGORITHM:

    This method implements the double-description version of the algorithm
    from [CPN16] with a vertical plane normal.

    REFERENCES:

    .. [CPN16] https://dx.doi.org/10.1109/TRO.2016.2623338
    """
    mass = 42.  # [kg]
    # mass has no effect on the output polygon, c.f. Section IV.C in
    # <https://hal.archives-ouvertes.fr/hal-01349880>
    # n = [0, 0, 1]
    crossmat_n = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])
    z_in, z_out = com[2], plane[2]

    G = contact_set.compute_grasp_matrix([0, 0, 0])
    F = -contact_set.compute_stacked_wrench_faces()
    b = zeros((F.shape[0], 1))
    # the input [b, -F] to cdd.Matrix represents (b - F x >= 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    M = cdd.Matrix(hstack([b, -F]), number_type='float')
    M.rep_type = cdd.RepType.INEQUALITY

    B = vstack([
        hstack([z_in * eye(3), crossmat_n]),
        hstack([zeros(3), com])])  # hstack([-(cross(n, p_in)), n])])
    C = 1. / (- mass * 9.81) * dot(B, G)
    d = hstack([com, [0]])
    # the input [d, -C] to cdd.Matrix.extend represents (d - C x == 0)
    # see ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
    M.extend(hstack([d.reshape((4, 1)), -C]), linear=True)

    # Convert from H- to V-representation
    # M.canonicalize()
    P = cdd.Polyhedron(M)
    V = array(P.get_generators())

    # Project output wrenches to 2D set
    vertices, rays = [], []
    for i in xrange(V.shape[0]):
        f_gi = dot(G, V[i, 1:])[:3]
        if V[i, 0] == 1:  # 1 = vertex, 0 = ray
            p_out = (z_out - z_in) * f_gi / (- mass * 9.81) + com
            vertices.append(p_out)
        else:
            r_out = (z_out - z_in) * f_gi / (- mass * 9.81)
            rays.append(r_out)
    return vertices, rays
