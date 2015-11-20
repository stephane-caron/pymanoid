#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of pymanoid.
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

"""

Compute the Gravito-Inertial Wrench Cone (GIWC) for a set of rectangular contact
surfaces positionned anywhere in space. The stability condition is that the
instantaneous gravito-inertial wrench belongs to the GIWC. See the paper at
<https://scaron.info/research/rss-2015.html> for details.

The calculation method here starts from friction cones located at the corners of
each rectangular surface.

"""

import openravepy
import pymanoid

from pymanoid.cone_duality import face_of_span, span_of_face
from numpy import array, dot, zeros
from scipy.linalg import block_diag

X = 0.3   # half-length of contacting links
Y = 0.1   # half-width of contacting links
mu = 0.6  # friction coefficient

CFC = array([  # Contact Force Cone
    # fx  fy    fz
    [-1,   0,  -mu],
    [+1,   0,  -mu],
    [0,   -1,  -mu],
    [0,   +1,  -mu]])

contacts = []


def add_contact(env, pos, rpy):
    print "Contact: POS=%s, RPY=%s" % (str(pos), str(rpy))
    mesh = pymanoid.Box(env, X=X, Y=Y, Z=0.01, color='r', pos=pos, rpy=rpy)
    contacts.append(mesh)
    return mesh


def calculate_local_forces_to_gi_matrix():
    A = zeros((6, 3 * 4 * len(contacts)))
    sign_pairs = [(-1, -1), (-1, +1), (+1, -1), (+1, +1)]
    for i, contact in enumerate(contacts):
        for (j, (sx, sy)) in enumerate(sign_pairs):
            x, y, z, o = dot(contact.T, [sx * X, sy * Y, 0, 1])
            Ai = array([
                [1, 0, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, -z, y],
                [z, 0, -x],
                [-y, x, 0]])
            A[0:6, (12 * i + 3 * j):(12 * i + 3 * (j + 1))] = -Ai
    return A


def compute_giwc_from_forces():
    A = calculate_local_forces_to_gi_matrix()
    CFC_all = block_diag(*[dot(CFC, block_diag(contact.R.T))
                           for contact in contacts for _ in xrange(4)])
    S = span_of_face(CFC_all)
    F = face_of_span(dot(A, S))
    return F


if __name__ == '__main__':
    env = openravepy.Environment()
    add_contact(env, pos=[0., .3, 0.], rpy=[0., 0., 0.])
    add_contact(env, pos=[0., -.5, 0.], rpy=[0., 0., 0.])
    GIWC = compute_giwc_from_forces()

    print "Gravito-Inertial Wrench Cone:"
    print GIWC
