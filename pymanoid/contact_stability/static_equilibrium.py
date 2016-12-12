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

from numpy import array, hstack, vstack, zeros

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/..')

from polyhedra.projection import project_polytope_bretl
from polyhedra.projection import project_polytope_cdd
from polyhedra.polygon import compute_polar_polygon


def compute_sep_lp(contact_set):
    p = [0, 0, 0]  # point where contact wrench is taken at
    G = contact_set.compute_grasp_matrix(p)
    F = contact_set.compute_stacked_wrench_faces()
    mass = 42.  # [kg]
    # mass has no effect on the output polygon, see Section IV.B in
    # <https://hal.archives-ouvertes.fr/hal-01349880> for details
    A = F
    b = zeros(A.shape[0])
    C = G[(0, 1, 2, 5), :]
    d = array([0, 0, mass * 9.81, 0])
    E = 1. / (mass * 9.81) * vstack([-G[4, :], +G[3, :]])
    f = array([p[0], p[1]])
    return A, b, C, d, E, f


def compute_sep_bretl(contact_set, solver='glpk'):
    """
    Compute the static-equilibrium polygon of the COM using Bretl & Lall's
    projection method.

    INPUT:

    - ``contact_set`` -- a ContactSet instance
    - ``solver`` -- (optional) LP backend for CVXOPT

    OUTPUT:

    List of vertices of the static-equilibrium polygon.

    ALGORITHM:

    This projection method is described in [BL08].

    REFERENCES:

    .. [BL08] https://dx.doi.org/10.1109/TRO.2008.2001360
    """
    A, b, C, d, E, f = compute_sep_lp(contact_set)
    vertices, _ = project_polytope_bretl(A, b, C, d, E, f, solver=solver)
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
    A, b, C, d, E, f = compute_sep_lp(contact_set)
    vertices, _ = project_polytope_cdd(A, b, C, d, E, f)
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
