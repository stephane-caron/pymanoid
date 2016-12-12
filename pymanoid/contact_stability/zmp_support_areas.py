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

import os
import sys

from numpy import array, dot, eye, hstack, vstack, zeros

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/..')

from polyhedra.projection import project_polytope_bretl
from polyhedra.projection import project_polytope_cdd


def compute_zmp_area_lp(contact_set, com, plane):
    z_com, z_zmp = com[2], plane[2]
    crossmat_n = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # n = [0, 0, 1]
    G = contact_set.compute_grasp_matrix([0, 0, 0])
    F = contact_set.compute_stacked_wrench_faces()
    mass = 42.  # [kg]
    # mass has no effect on the output polygon, c.f. Section IV.C in
    # <https://hal.archives-ouvertes.fr/hal-01349880>
    A = F
    b = zeros(A.shape[0])
    B = vstack([
        hstack([z_com * eye(3), crossmat_n]),
        hstack([zeros(3), com])])  # \sim hstack([-(cross(n, p_in)), n])])
    C = 1. / (mass * 9.81) * dot(B, G)
    d = hstack([com, [0]])
    E = (z_zmp - z_com) / (mass * 9.81) * G[:2, :]
    f = array([com[0], com[1]])
    return A, b, C, d, E, f


def compute_zmp_area_bretl(contact_set, com, plane, solver='glpk'):
    """
    Compute the pendular ZMP support area for a given COM position.

    INPUT:

    - ``contact_set`` -- a ContactSet instance
    - ``com`` -- COM position
    - ``plane`` -- position of horizontal plane
    - ``solver`` -- (optional) LP backend to CVXOPT

    OUTPUT:

    List of vertices of the area.

    ALGORITHM:

    This method relies on Bretl & Lall's projection method [BL08].

    REFERENCES:

    .. [BL08]  https://dx.doi.org/10.1109/TRO.2008.2001360
    """
    A, b, C, d, E, f = compute_zmp_area_lp(contact_set, com, plane)
    vertices, _ = project_polytope_bretl(A, b, C, d, E, f, solver=solver)
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
    A, b, C, d, E, f = compute_zmp_area_lp(contact_set, com, plane)
    vertices, _ = project_polytope_cdd(A, b, C, d, E, f)
    return vertices
