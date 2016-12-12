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

import numpy
import simplejson
import uuid

from numpy import array, cross, dot, eye, hstack, sqrt, vstack, zeros
from scipy.linalg import block_diag
from warnings import warn

from body import Box
from draw import draw_force
from optim import solve_relaxed_qp
from polyhedra import Cone, Cone3D
from rotations import crossmat
from sim import get_openrave_env

from contact_stability import \
    compute_sep_bretl as _compute_sep_bretl, \
    compute_sep_cdd as _compute_sep_cdd, \
    compute_sep_hull as _compute_sep_hull, \
    compute_zmp_area_bretl as _compute_zmp_area_bretl, \
    compute_zmp_area_cdd as _compute_zmp_area_cdd


class Contact(Box):

    THICKNESS = 0.01

    def __init__(self, X, Y, pos=None, rpy=None, friction=None,
                 pose=None, visible=True, name=None):
        """
        Create a new rectangular contact.

        INPUT:

        - ``X`` -- half-length of the contact surface
        - ``Y`` -- half-width of the contact surface
        - ``pos`` -- contact position in world frame
        - ``rpy`` -- contact orientation in world frame
        - ``friction`` -- friction coefficient
        - ``pose`` -- initial pose (supersedes pos and rpy)
        - ``visible`` -- initial box visibility
        """
        if not name:
            name = "Contact-%s" % str(uuid.uuid1())[0:3]
        self.friction = friction
        super(Contact, self).__init__(
            X, Y, Z=self.THICKNESS, pos=pos, rpy=rpy, pose=pose,
            visible=visible, dZ=-self.THICKNESS, name=name)

    """
    Geometry
    ========
    """

    @property
    def vertices(self):
        """Vertices of the contact area."""
        c1 = dot(self.T, array([+self.X, +self.Y, -self.Z, 1.]))[:3]
        c2 = dot(self.T, array([+self.X, -self.Y, -self.Z, 1.]))[:3]
        c3 = dot(self.T, array([-self.X, -self.Y, -self.Z, 1.]))[:3]
        c4 = dot(self.T, array([-self.X, +self.Y, -self.Z, 1.]))[:3]
        return [c1, c2, c3, c4]

    """
    Force Friction Cone
    ===================

    All linearized friction cones in pymanoid use the inner (conservative)
    approximation. See <https://scaron.info/teaching/friction-model.html>
    """

    @property
    def force_cone(self):
        """
        Contact-force friction cone.
        """
        return Cone3D(face=self.force_face, rays=self.force_rays)

    @property
    def force_face(self):
        """
        Face (H-rep) of the contact-force friction cone in world frame.
        """
        mu = self.friction / sqrt(2)  # inner approximation
        local_cone = array([
            [-1, 0, -mu],
            [+1, 0, -mu],
            [0, -1, -mu],
            [0, +1, -mu]])
        return dot(local_cone, self.R.T)

    @property
    def force_rays(self):
        """
        Rays (V-rep) of the contact-force friction cone in world frame.
        """
        mu = self.friction / sqrt(2)  # inner approximation
        f1 = dot(self.R, [+mu, +mu, +1])
        f2 = dot(self.R, [+mu, -mu, +1])
        f3 = dot(self.R, [-mu, +mu, +1])
        f4 = dot(self.R, [-mu, -mu, +1])
        return [f1, f2, f3, f4]

    @property
    def force_span(self):
        """
        Span matrix of the contact-force friction cone in world frame.
        """
        S = array(self.force_rays).T
        return S

    """
    Wrench Friction Cone
    ====================
    """

    @property
    def wrench_cone(self):
        """
        Contact-wrench friction cone.
        """
        wrench_cone = Cone(face=self.wrench_face, rays=self.wrench_rays)
        return wrench_cone

    @property
    def wrench_face(self):
        """
        Compute the matrix F of friction inequalities derived in [Caron2015]_.

        This matrix describes the linearized Coulomb friction model by:

            F * w <= 0

        where w is the contact wrench at the contact point (self.p) in the
        world frame.

        REFERENCES:

        .. [Caron2015] S. Caron, Q.-C. Pham, Y. Nakamura. Stability of Surface
           Contacts for Humanoid Robots Closed-Form Formulae of the Contact
           Wrench Cone for Rectangular Support Areas. ICRA 2015.

        """
        mu = self.friction / sqrt(2)  # inner approximation
        X, Y = self.X, self.Y
        local_cone = array([
            # fx fy             fz taux tauy tauz
            [-1,  0,           -mu,   0,   0,   0],
            [+1,  0,           -mu,   0,   0,   0],
            [0,  -1,           -mu,   0,   0,   0],
            [0,  +1,           -mu,   0,   0,   0],
            [0,   0,            -Y,  -1,   0,   0],
            [0,   0,            -Y,  +1,   0,   0],
            [0,   0,            -X,   0,  -1,   0],
            [0,   0,            -X,   0,  +1,   0],
            [-Y, -X, -(X + Y) * mu, +mu, +mu,  -1],
            [-Y, +X, -(X + Y) * mu, +mu, -mu,  -1],
            [+Y, -X, -(X + Y) * mu, -mu, +mu,  -1],
            [+Y, +X, -(X + Y) * mu, -mu, -mu,  -1],
            [+Y, +X, -(X + Y) * mu, +mu, +mu,  +1],
            [+Y, -X, -(X + Y) * mu, +mu, -mu,  +1],
            [-Y, +X, -(X + Y) * mu, -mu, +mu,  +1],
            [-Y, -X, -(X + Y) * mu, -mu, -mu,  +1]])
        return dot(local_cone, block_diag(self.R.T, self.R.T))

    @property
    def wrench_rays(self):
        """
        Rays (V-rep) of the contact wrench cone in world frame.
        """
        rays = []
        for v in self.vertices:
            x, y, z = v - self.p
            for f in self.force_rays:
                rays.append(hstack([f, cross(v - self.p, f)]))
        return rays

    @property
    def wrench_span(self):
        """
        Span matrix of the contact wrench cone in world frame.

        This matrix is such that all valid contact wrenches can be written as:

            w = S * lambda,     lambda >= 0

        where S is the friction span and lambda is a vector with positive
        coordinates. Note that the contact wrench w is taken at the contact
        point (self.p) and in the world frame.
        """
        span_blocks = []
        for (i, v) in enumerate(self.vertices):
            x, y, z = v - self.p
            Gi = vstack([eye(3), crossmat(v - self.p)])
            span_blocks.append(dot(Gi, self.force_span))
        S = hstack(span_blocks)
        assert S.shape == (6, 16)
        return S

    """
    Others
    ======
    """

    def draw_force_lines(self, length=0.25):
        env = get_openrave_env()
        handles = []
        for c in self.vertices:
            color = [0.1, 0.1, 0.1]
            color[numpy.random.randint(3)] += 0.2
            for f in self.force_rays:
                handles.append(env.drawlinelist(
                    array([c, c + length * f]),
                    linewidth=1, colors=color))
            handles.append(env.drawlinelist(
                array([c, c + length * self.n]),
                linewidth=5, colors=color))
        return handles

    @property
    def dict_repr(self):
        return {
            'X': self.X,
            'Y': self.Y,
            'Z': self.Z,
            'pos': list(self.p),
            'rpy': list(self.rpy),
            'friction': self.friction,
            'visible': self.is_visible,
        }

    def grasp_matrix(self, p):
        """
        Compute the grasp matrix from contact point to ``p`` in the world frame.

        INPUT:

        - ``p`` -- end point where the resultant wrench is taken

        OUTPUT:

        The grasp matrix G(p) converting the local contact wrench w to the
        contact wrench w(p) at another point p:

            w(p) = G(p) * w

        All wrenches are expressed with respect to the world frame.
        """
        x, y, z = self.p - p
        return array([
            # fx fy  fz taux tauy tauz
            [1,   0,  0,   0,   0,   0],
            [0,   1,  0,   0,   0,   0],
            [0,   0,  1,   0,   0,   0],
            [0,  -z,  y,   1,   0,   0],
            [z,   0, -x,   0,   1,   0],
            [-y,  x,  0,   0,   0,   1]])


class ContactSet(object):

    def __init__(self, contacts=None):
        """
        Create new contact set.

        INPUT:

        - ``contacts`` -- list or dictionary of Contact objects
        """
        if type(contacts) is list:
            self.contact_dict = {
                "Contact-%d" % i: c for (i, c) in enumerate(contacts)}
        elif type(contacts) is dict:
            self.contact_dict = contacts
        else:  # contacts is None
            assert contacts is None
            self.contact_dict = {}

    @property
    def nb_contacts(self):
        return len(self.contact_dict)

    @staticmethod
    def from_json(path):
        with open(path, 'r') as fp:
            d = simplejson.load(fp)
        contacts = {
            name: Contact(**kwargs) for (name, kwargs) in d.iteritems()}
        return ContactSet(contacts)

    def save_json(self, path):
        d = {name: contact.dict_repr
             for (name, contact) in self.contact_dict.iteritems()}
        with open(path, 'w') as fp:
            simplejson.dump(d, fp, indent=4, sort_keys=True)

    def __contains__(self, name):
        """Check whether a named contact is present."""
        return name in self.contact_dict

    def __getitem__(self, name):
        """Get a named contact from the set."""
        return self.contact_dict[name]

    def __iter__(self):
        for contact in self.contact_dict.itervalues():
            yield contact

    @property
    def contacts(self):
        """Iterate contacts in the set."""
        for contact in self.contact_dict.itervalues():
            yield contact

    def find_supporting_forces(self, wrench, point, friction_weight=.1,
                               pressure_weight=10.):
        """
        Find a set of contact forces supporting a given wrench.

        If the resultant wrench ``wrench`` (expressed at ``point``) can be
        supported by the contact set, output a set of supporting contact
        forces that minimizes the cost

            sum_{contact i}  w_t * |f_{i,t}|^2 + w_z * |f_{i,z}|^2

        where |f_{i,t}| (resp. f_{i,z}) is the norm of the i-th friction (resp.
        pressure) force.

        INPUT:

        - ``wrench`` -- the resultant wrench to be realized
        - ``point`` -- point where the wrench is expressed
        - ``friction_weight`` -- weight for friction term in optim. objective
        - ``pressure_weight`` -- weight for pressure term in optim. objective

        OUTPUT:

        A list of couples (contact point, contact force) expressed in the world
        frame.

        .. NOTE::

            Physically, contact results in continuous distributions of friction
            and pressure forces. However, one can model them without loss of
            generality (in terms of the resultant wrench) by considering only
            point contact forces applied at the vertices of the contact area.
            See [CPN]_ for details.

        REFERENCES:

        .. [CPN] Caron, Pham, Nakamura, "Stability of surface contacts for
           humanoid robots: Closed-form formulae of the contact wrench cone for
           rectangular support areas." 2015 IEEE International Conference on
           Robotics and Automation (ICRA).
        """
        n = 12 * self.nb_contacts
        nb_forces = n / 3
        P_fric = block_diag(*[
            array([
                [1., 0., 0.],
                [0., 1., 0.],
                [0., 0., 0.]])
            for _ in xrange(nb_forces)])
        P_press = block_diag(*[
            array([
                [0., 0., 0.],
                [0., 0., 0.],
                [0., 0., 1.]])
            for _ in xrange(nb_forces)])
        o_z = hstack([
            [0, 0, 1. / n]
            for _ in xrange(nb_forces)])
        P_press -= dot(o_z.reshape((n, 1)), o_z.reshape((1, n)))
        P_local = friction_weight * P_fric + pressure_weight * P_press
        RT_diag = block_diag(*[
            contact.R.T
            for contact in self.contacts for _ in xrange(4)])
        P = dot(RT_diag.T, dot(P_local, RT_diag))
        q = zeros((n,))
        G = self.compute_stacked_force_faces()
        h = zeros((G.shape[0],))  # G * x <= h
        A = self.compute_grasp_matrix_from_forces(point)
        b = wrench
        # f_all = cvxopt_solve_qp(P, q, G, h, A, b)  # useful for debugging
        f_all = solve_relaxed_qp(P, q, G, h, A, b, tol=1e-2)
        if f_all is None:
            return None
        output, next_index = [], 0
        for i, contact in enumerate(self.contacts):
            for j, p in enumerate(contact.vertices):
                output.append((p, f_all[next_index:next_index + 3]))
                next_index += 3
        return output

    def draw_supporting_forces(self, wrench, point, force_scale=0.0025,
                               **kwargs):
        support = self.find_supporting_forces(wrench, point, **kwargs)
        if support is None:
            warn("draw_supporting_support(): there are no supporting forces")
            return
        handles = [draw_force(c, fc, force_scale) for (c, fc) in support]
        return handles

    def find_static_supporting_forces(self, com, mass):
        """
        Find a set of contact forces supporting the robot in static equilibrium
        when its center of mass is located at ``com``.

        INPUT:

        - ``com`` -- position of the center of mass
        - ``mass`` -- total mass of the robot

        OUTPUT:

        A list of couples (contact point, contact force) expressed in the world
        frame.

        .. SEEALSO::

            :meth:`pymanoid.contact.ContactSet.find_supporting_forces`,
        """
        f = numpy.array([0., 0., mass * 9.81])
        tau = zeros(3)
        wrench = numpy.hstack([f, tau])
        return self.find_supporting_forces(wrench, com)

    def find_supporting_wrenches(self, wrench, point, friction_weight=.1,
                                 pressure_weight=10.):
        n = 6 * self.nb_contacts
        P = eye(n)
        q = zeros((n,))
        G = self.compute_stacked_wrench_faces()
        h = zeros((G.shape[0],))  # G * x <= h
        A = self.compute_grasp_matrix(point)
        b = wrench
        w_all = solve_relaxed_qp(P, q, G, h, A, b, tol=1e-2)
        if w_all is None:
            return None
        output, next_index = [], 0
        for i, contact in enumerate(self.contacts):
            for j, p in enumerate(contact.vertices):
                output.append((p, w_all[next_index:next_index + 6]))
                next_index += 6
        return output

    def is_inside_static_equ_polygon(self, com, mass):
        """
        Check whether a given COM position lies inside the static-equilibrium
        COM polygon.

        INPUT:

        - ``com`` -- COM position to check
        - ``mass`` -- total mass of the robot

        OUTPUT:

        True if and only if ``com`` is inside the static-equilibrium polygon.
        """
        return self.find_static_supporting_forces(com, mass) is not None

    def compute_stacked_force_faces(self):
        """
        Compute the friction constraints on all contact forces.

        The friction matrix F is defined so that friction constraints on all
        contact wrenches are written:

            F * f_all <= 0

        where f_all is the stacked vector of contact forces, each taken at its
        corresponding contact point in the world frame.
        """
        return block_diag(*[c.force_face for c in self.contacts
                            for p in c.vertices])

    def compute_stacked_wrench_faces(self):
        """
        Compute the friction constraints on all contact wrenches.

        The friction matrix F is defined so that friction constraints on all
        contact wrenches are written:

            F * w_all <= 0

        where w_all is the stacked vector of contact wrenches, each taken at its
        corresponding contact point in the world frame.
        """
        return block_diag(*[c.wrench_face for c in self.contacts])

    def compute_wrench_span(self, p):
        """
        Compute the span matrix of the contact wrench cone in world frame.

        INPUT:

        - ``p`` -- point where the resultant wrench is taken at

        OUTPUT:

        The span matrix S(p) such that all valid contact wrenches can be written
        as:

            w(p) = S(p) * lambda,     lambda >= 0

        where w(p) is the contact wrench with respect to point p, lambda is a
        vector with positive coordinates.
        """
        span_blocks = []
        for contact in self.contacts:
            x, y, z = contact.p - p
            Gi = array([
                [1,  0,  0, 0, 0, 0],
                [0,  1,  0, 0, 0, 0],
                [0,  0,  1, 0, 0, 0],
                [0, -z,  y, 1, 0, 0],
                [z,  0, -x, 0, 1, 0],
                [-y, x,  0, 0, 0, 1]])
            span_blocks.append(dot(Gi, contact.wrench_span))
        S = hstack(span_blocks)
        assert S.shape == (6, 16 * self.nb_contacts)
        return S

    def compute_wrench_face(self, p):
        """
        Compute the face matrix of the contact wrench cone in the world frame.

        INPUT:

        - ``p`` -- point where the resultant wrench is taken at

        OUTPUT:

        The friction matrix F(p) such that all valid contact wrenches satisfy:

            F(p) * w(p) <= 0,

        where w(p) is the resultant contact wrench at p.
        """
        S = self.compute_wrench_span(p)
        return Cone.face_of_span(S)

    def compute_grasp_matrix(self, p):
        """
        Compute the grasp matrix of all contact wrenches at point p.

        INPUT:

        - ``p`` -- point where to take the resultant wrench

        OUTPUT:

        The grasp matrix G(p) giving the resultant contact wrench w(p) of all
        contact wrenches by:

            w(p) = G(p) * w_all,

        with w_all the stacked vector of contact wrenches, each wrench being
        taken at its respective contact point and in the world frame.
        """
        return hstack([c.grasp_matrix(p) for c in self.contacts])

    def compute_grasp_matrix_from_forces(self, p):
        """
        Compute the grasp matrix from all contact points in the set.

        INPUT:

        - ``p`` -- point where to take the resultant wrench

        OUTPUT:

        The grasp matrix G(p) giving the resultant contact wrench w(p) of all
        contact forces by:

            w(p) = G(p) * f_all,

        with f_all the stacked vector of contact forces, each force being
        taken at its respective contact point.
        """
        G = zeros((6, 3 * 4 * self.nb_contacts))
        for i, contact in enumerate(self.contacts):
            for j, cp in enumerate(contact.vertices):
                x, y, z = cp - p
                Gi = array([
                    [1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1],
                    [0, -z, y],
                    [z, 0, -x],
                    [-y, x, 0]])
                G[:, (12 * i + 3 * j):(12 * i + 3 * (j + 1))] = Gi
        return G

    def compute_static_equilibrium_polygon(self, method='hull'):
        """
        Compute the static-equilibrium polygon of the center of mass.

        INPUT:

        - ``method`` -- (optional) choice between 'bretl', 'cdd' or 'hull'

        OUTPUT:

        List of vertices of the static-equilibrium polygon.

        ALGORITHM:

        The method 'bretl' is adapted from in [BL08] where the
        static-equilibrium polygon was introduced. The method 'cdd' corresponds
        to the double-description approach described in [CPN16]. See the
        Appendix from [CK16] for a performance comparison.

        REFERENCES:

        .. [BL08]  https://dx.doi.org/10.1109/TRO.2008.2001360
        .. [CPN16] https://dx.doi.org/10.1109/TRO.2016.2623338
        .. [CK16]  https://hal.archives-ouvertes.fr/hal-01349880
        """
        if method == 'bretl':
            return _compute_sep_bretl(self)
        elif method == 'cdd':
            return _compute_sep_cdd(self)
        elif method == 'hull':
            return _compute_sep_hull(self)
        return Exception("invalid ``method`` argument")

    def compute_zmp_support_area(self, com, plane, method='bretl'):
        """
        Compute the (pendular) ZMP support area for a given COM position.

        INPUT:

        - ``com`` -- COM position
        - ``plane`` -- origin (in world frame) of the virtual plane
        - ``method`` -- (optional) choice between 'bretl' or 'cdd'

        OUTPUT:

        List of vertices of the ZMP support area.

        ALGORITHM:

        The method 'bretl' is adapted from in [BL08] where the
        static-equilibrium polygon was introduced. The method 'cdd' corresponds
        to the double-description approach described in [CPN16]. See the
        Appendix from [CK16] for a performance comparison.

        REFERENCES:

        .. [BL08]  https://dx.doi.org/10.1109/TRO.2008.2001360
        .. [CPN16] https://dx.doi.org/10.1109/TRO.2016.2623338
        .. [CK16]  https://hal.archives-ouvertes.fr/hal-01349880
        """
        if method == 'cdd':
            return _compute_zmp_area_cdd(self)
        elif method == 'bretl':
            return _compute_zmp_area_bretl(self)
        return Exception("invalid ``method`` argument")
