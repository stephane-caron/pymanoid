#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of openravepypy.
#
# openravepypy is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# openravepypy is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openravepypy. If not, see <http://www.gnu.org/licenses/>.

"""

Compute the Gravito-Inertial Wrench Cone (GIWC) for a set of rectangular contact
surfaces positionned anywhere in space. The stability condition is that the
instantaneous gravito-inertial wrench belongs to the GIWC. See the paper at
<https://scaron.info/research/rss-2015.html> for details.

The calculation method here uses one Contact Wrench Cone (CWC) for each
rectangular contact. See <https://scaron.info/research/icra-2015.html> for the
derivation of the CWC.

"""

import openravepy
import openravepypy

from openravepypy.cone_duality import face_of_span, span_of_face
from numpy import array, dot, zeros
from scipy.linalg import block_diag

X = 0.3   # half-length of contacting links
Y = 0.1   # half-width of contacting links
mu = 0.6  # friction coefficient

CWC = array([  # Contact Wrench Cone
    # fx  fy              fz  taux tauy tauz
    [-1,   0,            -mu,    0,   0,   0],
    [+1,   0,            -mu,    0,   0,   0],
    [0,   -1,            -mu,    0,   0,   0],
    [0,   +1,            -mu,    0,   0,   0],
    [0,    0,             -Y,   -1,   0,   0],
    [0,    0,             -Y,   +1,   0,   0],
    [0,    0,             -X,    0,  -1,   0],
    [0,    0,             -X,    0,  +1,   0],
    [-Y,  -X,  -(X + Y) * mu,  +mu,  +mu,  -1],
    [-Y,  +X,  -(X + Y) * mu,  +mu,  -mu,  -1],
    [+Y,  -X,  -(X + Y) * mu,  -mu,  +mu,  -1],
    [+Y,  +X,  -(X + Y) * mu,  -mu,  -mu,  -1],
    [+Y,  +X,  -(X + Y) * mu,  +mu,  +mu,  +1],
    [+Y,  -X,  -(X + Y) * mu,  +mu,  -mu,  +1],
    [-Y,  +X,  -(X + Y) * mu,  -mu,  +mu,  +1],
    [-Y,  -X,  -(X + Y) * mu,  -mu,  -mu,  +1]])


contacts = []


def add_contact(env, pos, rpy):
    print "Contact: POS=%s, RPY=%s" % (str(pos), str(rpy))
    mesh = openravepypy.Box(env, X=X, Y=Y, Z=0.01, color='r', pos=pos, rpy=rpy)
    contacts.append(mesh)
    return mesh


def calculate_local_wrench_to_gi_matrix():
    """
    Return the matrix A mapping local contact wrenches w_i (expressed in the
    *world* frame) to the gravito-inertial wrench.
    """
    A = zeros((6, 6 * len(contacts)))
    for i, contact in enumerate(contacts):
        x, y, z = contact.p
        Ai = array([
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, -z, y, 1, 0, 0],
            [z, 0, -x, 0, 1, 0],
            [-y, x, 0, 0, 0, 1]])
        A[0:6, (6 * i):(6 * (i + 1))] = -Ai
    return A


def compute_giwc_from_wrenches(full=True):
    """
    Compute Gravito-Inertial Wrench Cone (GIWC) from Contact Wrench Cones
    (CWCs).
    """
    global CWC_all
    A = calculate_local_wrench_to_gi_matrix()
    # right vector of CWC_all is the stacked vector w_all of
    # contact wrenches in the *world* frame
    CWC_all = block_diag(*[
        dot(CWC, block_diag(contact.R.T, contact.R.T))
        for contact in contacts])
    S = span_of_face(CWC_all)
    F = face_of_span(dot(A, S))
    return F


if __name__ == '__main__':
    env = openravepy.Environment()
    add_contact(env, pos=[0., .3, 0.], rpy=[0., 0., 0.])
    add_contact(env, pos=[0., -.5, 0.], rpy=[0., 0., 0.])
    GIWC = compute_giwc_from_wrenches()

    print "Gravito-Inertial Wrench Cone:"
    print GIWC
