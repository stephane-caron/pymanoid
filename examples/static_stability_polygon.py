#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
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

import IPython
import cdd
import openravepy
import pylab
import pymanoid
import thread
import time

from numpy import array, cross, dot, eye, hstack, vstack, zeros
from pymanoid.toolbox import norm
from scipy.linalg import block_diag


foot_half_height = 0.01  # [m]
foot_half_length = 0.11  # [m]
foot_half_width = 0.07   # [m]

init_left_foot_pos = [0.68546862, -0.20824458, 0.32014298]
init_left_foot_rpy = [0.11566877, 0.26659006, -3.11573395]
init_right_foot_pos = [-0.37655576, 0.24925528, 0.36054255]
init_right_foot_rpy = [0.19088011, -0.20956477, -0.0788266]


class Contact(pymanoid.Box):

    def __init__(self, env, pos, rpy, fric_coeff):
        self.env = env
        self.fric_coeff = fric_coeff
        super(Contact, self).__init__(
            env, X=foot_half_length, Y=foot_half_width, Z=foot_half_height,
            color='r', pos=pos, rpy=rpy)

    @property
    def T(self):
        T = super(Contact, self).T
        n = T[0:3, 2]
        T[0:3, 3] += self.Z * n
        return T

    @property
    def pose(self):
        pose = super(Contact, self).pose
        pose[4:] += self.Z * self.n   # self.n calls self.T
        return pose

    @property
    def contact_points(self):
        c1 = dot(self.T, array([+self.X, +self.Y, -self.Z, 1.]))[:3]
        c2 = dot(self.T, array([+self.X, -self.Y, -self.Z, 1.]))[:3]
        c3 = dot(self.T, array([-self.X, -self.Y, -self.Z, 1.]))[:3]
        c4 = dot(self.T, array([-self.X, +self.Y, -self.Z, 1.]))[:3]
        return [c1, c2, c3, c4]

    @property
    def gaw_face_local(self):
        """Ground-applied wrench cone, H-representation, local frame."""
        X, Y = self.X, self.Y
        mu = self.fric_coeff
        grw_face = array([  # Ground Reaction Wrench Cone
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
        gaw_face = grw_face
        gaw_face[:, (2, 3, 4)] *= -1  # oppose local Z-axis
        return gaw_face

    @property
    def gaw_face_world(self):
        """Ground-applied wrench cone, H-representation, world frame."""
        return dot(self.gaw_face_local, block_diag(self.R.T, self.R.T))


def project_cdd_output(V, D):
    origin = array([0., 0., 0.])
    normal = array([0., 0., 1.])
    assert norm(cross(normal, [0., 0., 1.])) < 1e-10
    vertices, rays = [], []
    for i in xrange(V.shape[0]):
        if V[i, 0] == 1:  # 1 = vertex, 0 = ray
            p = dot(D, V[i, 1:])
            vertices.append([p[0], p[1], origin[2]])
        else:
            r = dot(D, V[i, 1:])
            rays.append([r[0], r[1], origin[2]])
    return vertices, rays


class ContactSet(object):

    def __init__(self, env, mass=10.):
        self.contacts = []
        self.env = env
        self.mass = mass

    @property
    def nb_contacts(self):
        return len(self.contacts)

    def add(self, pos, rpy, fric_coeff=0.5):
        self.contacts.append(Contact(self.env, pos, rpy, fric_coeff))

    def compute_gaw_to_gi_matrix(self):
        """
        Compute the conversion matrix from ground-applied wrenches to the
        gravito-inertial wrench.
        """
        N_gi = zeros((6, 6 * self.nb_contacts))
        for i, contact in enumerate(self.contacts):
            x, y, z = contact.p
            Ni = array([
                [1, 0, 0, 0, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [0, 0, 1, 0, 0, 0],
                [0, -z, y, 1, 0, 0],
                [z, 0, -x, 0, 1, 0],
                [-y, x, 0, 0, 0, 1]])
            N_gi[:, (6 * i):(6 * (i + 1))] = Ni
        return N_gi

    def compute_static_stability_area(self, debug=True):
        t0 = time.time()
        W_gi = self.compute_gaw_to_gi_matrix()

        # Inequalities:  A [GAW_1 GAW_2 ...] <= 0
        A = block_diag(*[c.gaw_face_world for c in self.contacts])
        b = zeros((A.shape[0], 1))

        # Equalities:  C [GAW_1 GAW_2 ...] + d == 0
        C = W_gi[(0, 1, 2, 5), :]
        d = -array([0, 0, -self.mass * 9.81, 0])

        # H-representation: b - A x >= 0
        # ftp://ftp.ifor.math.ethz.ch/pub/fukuda/cdd/cddlibman/node3.html
        # input to cdd.Matrix is [b, -A]
        M = cdd.Matrix(hstack([b, -A]), number_type='float')
        M.rep_type = cdd.RepType.INEQUALITY
        M.extend(hstack([d.reshape((4, 1)), C]), linear=True)
        P = cdd.Polyhedron(M)
        V = array(P.get_generators())
        if V.shape[0] < 1:
            return []

        if debug:
            assert all(dot(A, V[1, 1:]) < 1e-2), dot(A, V[1, 1:])
            assert norm(dot(C, V[1, 1:]) + d) < 1e-5

        # COM position from GAW:  [pGx, pGy] = D * [GAW_1 GAW_2 ...]
        D = 1. / (-self.mass * 9.81) * vstack([-W_gi[4, :], +W_gi[3, :]])
        vertices, _ = project_cdd_output(V, D)
        if debug:
            print "Static stability polygon (%d vertices) computed in %d ms" \
                % (len(vertices), int(1000 * (time.time() - t0)))
        return vertices

    def draw_static_stability_area(self, plot_type=6, alpha=0.5):
        vertices = self.compute_static_stability_area()
        self.com_polygon = vertices
        if not vertices:
            return []
        handle = pymanoid.draw_polygon(
            self.env, vertices, array([0., 0., 1.]),
            plot_type=plot_type, color=(0., 0.5, 0., alpha))
        return [handle]


def update_thread(dt=1e-2):
    T = [eye(4) for contact in contact_set.contacts]
    gui_handles = None
    while True:
        for i, contact in enumerate(contact_set.contacts):
            if pylab.norm(T[i] - contact.T) > 1e-4:
                #
                # NB: cdd will fail to instantiate polyhedra if transforms
                # contain small values such as 1e-8, so we round them off.
                #
                contact.set_transform(contact.T.round(5))
                new_handles = contact_set.draw_static_stability_area()
                gui_handles = new_handles
                T[i] = contact.T.copy()
        time.sleep(dt)
    return gui_handles   # avoid lint warning :p


if __name__ == '__main__':
    env = openravepy.Environment()
    env.SetViewer('qtcoin')
    viewer = env.GetViewer()
    viewer.SetBkgndColor([.7, .7, .9])
    viewer.SetCamera([
        [0, 0, -1, 3],
        [1, 0, 0, 0.1],
        [0, -1, 0, 0.5],
        [0, 0, 0, 1]])
    contact_set = ContactSet(env)
    contact_set.add(pos=init_left_foot_pos, rpy=init_left_foot_rpy)
    contact_set.add(pos=init_right_foot_pos, rpy=init_right_foot_rpy)
    thread.start_new(update_thread, ())
    IPython.embed()
