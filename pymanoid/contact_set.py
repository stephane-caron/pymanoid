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

from numpy import array, dot, eye, hstack, vstack, zeros
from scipy.linalg import block_diag
from warnings import warn

from draw import draw_force
from optim import solve_relaxed_qp
from polyhedra import Cone
from polyhedra.polygon import compute_polar_polygon
from polyhedra import PolytopeProjector


class ContactSet(object):

    def __init__(self, contacts=None):
        """
        Create a new contact set.

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
        if method == 'hull':
            A_O = self.compute_wrench_face([0, 0, 0])
            k, a_Oz, a_x, a_y = A_O.shape[0], A_O[:, 2], A_O[:, 3], A_O[:, 4]
            B, c = hstack([-a_y.reshape((k, 1)), +a_x.reshape((k, 1))]), -a_Oz
            return compute_polar_polygon(B, c)
        p = [0, 0, 0]  # point where contact wrench is taken at
        G = self.compute_grasp_matrix(p)
        F = self.compute_stacked_wrench_faces()
        mass = 42.  # [kg]
        # mass has no effect on the output polygon, see Section IV.B in
        # <https://hal.archives-ouvertes.fr/hal-01349880> for details
        pp = PolytopeProjector()
        pp.set_inequality(
            F,
            zeros(F.shape[0]))
        pp.set_equality(
            G[(0, 1, 2, 5), :],
            array([0, 0, mass * 9.81, 0]))
        pp.set_output(
            1. / (mass * 9.81) * vstack([-G[4, :], +G[3, :]]),
            array([p[0], p[1]]))
        return pp.project(method)

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
        z_com, z_zmp = com[2], plane[2]
        crossmat_n = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # n = [0, 0, 1]
        G = self.compute_grasp_matrix([0, 0, 0])
        F = self.compute_stacked_wrench_faces()
        mass = 42.  # [kg]
        # mass has no effect on the output polygon, c.f. Section IV.C in
        # <https://hal.archives-ouvertes.fr/hal-01349880>
        pp = PolytopeProjector()
        pp.set_inequality(
            F,
            zeros(F.shape[0]))
        B = vstack([
            hstack([z_com * eye(3), crossmat_n]),
            hstack([zeros(3), com])])  # \sim hstack([-(cross(n, p_in)), n])])
        pp.set_equality(
            1. / (mass * 9.81) * dot(B, G),
            hstack([com, [0]]))
        pp.set_output(
            (z_zmp - z_com) / (mass * 9.81) * G[:2, :],
            array([com[0], com[1]]))
        return pp.project(method)
