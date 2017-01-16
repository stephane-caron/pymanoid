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

from numpy import array, cross, dot, eye, hstack, vstack, zeros
from scipy.linalg import block_diag
from scipy.spatial.qhull import QhullError

from optim import solve_relaxed_qp
from optim import solve_qp
from polyhedra import Cone
from polyhedra.polygon import compute_polygon_hull
from polyhedra import Polytope, PolytopeProjector
from sim import gravity


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

    @property
    def nb_contacts(self):
        return len(self.contact_dict)

    def find_supporting_forces(self, wrench, point, friction_weight=.1,
                               pressure_weight=10.):
        """Find supporting forces for a given net wrench.

        If the net wrench ``wrench`` (expressed at ``point``) can be supported
        by the contact set, output a set of supporting contact forces that
        minimizes the cost

        .. math::

            \\sum_{\\textrm{contact }i}
            w_t \\|f_i^t\\|^2 + w_n \\|f_i^n\\|^2

        where :math:`f_i^t` (resp. :math:`f_i^n`) is the friction (resp.
        pressure) force at the :math:`i^\\mathrm{th}` contact.

        Parameters
        ----------
        wrench : ndarray
            Resultant wrench to be realized.
        point : ndarray
            Point where the wrench is expressed.
        friction_weight : double
            Weight :math:`w_t` for friction terms in the cost function.
            The default value is ``0.1``.
        pressure_weight : double
            Weight :math:`w_n` for pressure terms in the cost function.
            The default value is ``10``.

        Returns
        -------
        support : list of (3,) ndarray couples
            List of couples (contact point, contact force) with coordinates
            expressed in the world frame.

        Notes
        -----
        Physically, contact results in continuous distributions of friction and
        pressure forces. However, one can model them without loss of generality
        (in terms of the resultant wrench) by considering only point contact
        forces applied at the vertices of the contact area (see e.g. [CPN15]_)
        which is why we only consider point contacts here.

        References
        ----------
        .. [CPN15] Caron, Pham, Nakamura, "Stability of surface contacts for
            humanoid robots: Closed-form formulae of the contact wrench cone for
            rectangular support areas," 2015 IEEE International Conference on
            Robotics and Automation (ICRA).
            `[doi] <https://doi.org/10.1109/ICRA.2015.7139910>`__
            `[pdf] <https://scaron.info/papers/conf/caron-icra-2015.pdf>`__
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
        G = block_diag(*[
            c.force_face for c in self.contacts for p in c.vertices])
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

    def find_static_supporting_forces(self, com, mass):
        """Find supporting forces in static-equilibrium.

        Find a set of contact forces that support the robot in static
        equilibrium when its center of mass is located at ``com``.

        Parameters
        ----------
        com : ndarray
            Position of the center of mass.
        mass : double
            Total mass of the robot in [kg].

        Returns
        -------
        support : list of (Contact, ndarray) couples
            List of couples (contact, contact force) with coordinates expressed
            in the world frame.

        See Also
        --------
        - :meth:`pymanoid.contact.ContactSet.find_supporting_forces`
        - :meth:`pymanoid.contact.ContactSet.find_supporting_wrenches`
        """
        f = numpy.array([0., 0., mass * 9.81])
        tau = zeros(3)
        wrench = numpy.hstack([f, tau])
        return self.find_supporting_forces(wrench, com)

    def find_supporting_wrenches(self, wrench, point):
        n = 6 * self.nb_contacts
        P = eye(n)
        q = zeros((n,))
        G = block_diag(*[c.wrench_face for c in self.contacts])
        h = zeros((G.shape[0],))  # G * x <= h
        A = self.compute_grasp_matrix(point)
        b = wrench
        w_all = solve_qp(P, q, G, h, A, b)
        if w_all is None:
            return None
        return [(contact, w_all[6 * i:6 * (i + 1)])
                for i, contact in enumerate(self.contacts)]

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

        Parameters
        ----------
        method : string, optional
            choice between 'bretl', 'cdd' or 'hull'

        Returns
        -------
        vertices : list of ndarrays
            2D vertices of the static-equilibrium polygon.

        Notes
        -----
        The method 'bretl' is adapted from in [BL08]_ where the
        static-equilibrium polygon was introduced. The method 'cdd' corresponds
        to the double-description approach described in [CPN16]_. See the
        Appendix from [CK16]_ for a performance comparison.

        References
        ----------

        .. [BL08] T. Bretl and S. Lall, "Testing Static Equilibrium for Legged
            Robots," IEEE Transactions on Robotics, vol. 24, no. 4, pp. 794-807,
            Aug. 2008.
            `[doi] <https://dx.doi.org/10.1109/TRO.2008.2001360>`__

        .. [CPN16] Stéphane Caron, Quang-Cuong Pham and Yoshihiko Nakamura, "ZMP
            support areas for multi-contact mobility under frictional
            constraints," IEEE Transactions on Robotics, Dec. 2016.
            `[doi] <https://doi.org/10.1109/TRO.2016.2623338>`__
            `[pdf] <https://scaron.info/papers/journal/caron-tro-2016.pdf>`__

        .. [CK16] Stéphane Caron and Abderrahmane Kheddar, "Multi-contact
            Walking Pattern Generation based on Model Preview Control of 3D COM
            Accelerations," 2016 IEEE-RAS 16th International Conference on
            Humanoid Robots (Humanoids), Cancun, Mexico, 2016, pp. 550-557.
            `[doi] <https://doi.org/10.1109/HUMANOIDS.2016.7803329>`__
            `[pdf] <https://hal.archives-ouvertes.fr/hal-01349880>`__
        """
        if method == 'hull':
            A_O = self.compute_wrench_face([0, 0, 0])
            k, a_Oz, a_x, a_y = A_O.shape[0], A_O[:, 2], A_O[:, 3], A_O[:, 4]
            B, c = hstack([-a_y.reshape((k, 1)), +a_x.reshape((k, 1))]), -a_Oz
            return compute_polygon_hull(B, c)
        p = [0, 0, 0]  # point where contact wrench is taken at
        G = self.compute_grasp_matrix(p)
        F = block_diag(*[contact.wrench_face for contact in self.contacts])
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


        Parameters
        ----------
        com : ndarray
            COM position
        plane : ndarray
            origin (in world frame) of the virtual plane
        method : string, optional
            choice between ``"bretl"`` or ``"cdd"``

        Returns
        -------
        vertices : list of ndarrays
            List of vertices of the ZMP support area.

        Notes
        -----
        The method 'bretl' is adapted from in [BL08]_ where the
        static-equilibrium polygon was introduced. The method 'cdd' corresponds
        to the double-description approach described in [CPN16]_. See the
        Appendix from [CK16]_ for a performance comparison.
        """
        z_com, z_zmp = com[2], plane[2]
        crossmat_n = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # n = [0, 0, 1]
        G = self.compute_grasp_matrix([0, 0, 0])
        F = block_diag(*[contact.wrench_face for contact in self.contacts])
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

    def compute_pendular_accel_cone(self, com, zdd_max=None, reduced=False):
        """
        Compute the (pendular) COM acceleration cone for a given COM position.

        This pendular cone is the reduction of the Contact Wrench Cone when the
        angular momentum at the COM is zero.

        INPUT:

        - ``com`` -- COM position, or list of COM vertices
        - ``zdd_max`` -- (optional) maximum vertical acceleration in output cone
        - ``reduced`` -- (optional) if True, will return the 2D reduced form

        OUTPUT:

        List of 3D vertices of the (truncated) COM acceleration cone, or of the
        2D vertices of the reduced form if ``reduced`` is ``True``.

        When ``com`` is a list of vertices, the returned cone corresponds to COM
        accelerations that are feasible from *all* COM located inside the
        polytope. See [CK16]_ for details on this conservative criterion.

        ALGORITHM:

        The method is based on a rewriting of the cone formula, followed by a 2D
        convex hull on dual vertices. The algorithm is described in [CK16]_.
        """
        com_vertices = [com] if type(com) is not list else com
        CWC_O = self.compute_wrench_face([0., 0., 0.])
        B_list, c_list = [], []
        for (i, v) in enumerate(com_vertices):
            B = CWC_O[:, :3] + cross(CWC_O[:, 3:], v)
            c = dot(B, gravity)
            B_list.append(B)
            c_list.append(c)
        B = vstack(B_list)
        c = hstack(c_list)
        try:
            g = -gravity[2]  # gravity constant (positive)
            check = c / B[:, 2]
            assert max(check) - min(check) < 1e-10, \
                "max - min failed (%.1e)" % ((max(check) - min(check)))
            assert abs(check[0] - (-g)) < 1e-10, "check is not -g?"
            B_2d = hstack([B[:, j].reshape((B.shape[0], 1)) for j in [0, 1]])
            sigma = c / g  # see Equation (30) in [CK16]
            reduced_hull = compute_polygon_hull(B_2d, sigma)
            if reduced:
                return reduced_hull
            return self._expand_reduced_pendular_cone(reduced_hull, zdd_max)
        except QhullError:
            raise Exception("Cannot compute 2D polar for acceleration cone")

    @staticmethod
    def _expand_reduced_pendular_cone(reduced_hull, zdd_max=None):
        g = -gravity[2]  # gravity constant (positive)
        zdd = +g if zdd_max is None else zdd_max
        vertices_at_zdd = [
            array([a * (g + zdd), b * (g + zdd), zdd])
            for (a, b) in reduced_hull]
        return Polytope(vertices=[gravity] + vertices_at_zdd)
