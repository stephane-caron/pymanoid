#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import array, cross, dot, eye, hstack, sqrt, vstack, zeros
from scipy.linalg import block_diag
from scipy.spatial.qhull import QhullError

from body import Box
from optim import solve_qp
from polygons import compute_polygon_hull
from polyhedra import compute_cone_face_matrix, project_polytope
from rotations import crossmat
from sim import gravity


class Contact(Box):

    """
    Rectangular contact surface.

    Parameters
    ----------
    shape : (scalar, scalar)
        Surface dimensions (half-length, half-width) in [m].
    pos : ndarray
        Contact position in world frame.
    rpy : ndarray
        Contact orientation in world frame.
    pose : ndarray
        Initial pose. Supersedes ``pos`` and ``rpy`` if they are provided at the
        same time.
    friction : scalar
        Static friction coefficient.
    visible : bool, optional
        Initial visibility. Defaults to ``True``.
    name : string, optional
        Name in OpenRAVE scope.
    """

    THICKNESS = 0.01
    """Thickness in [m] of the contact patch."""

    def __init__(self, shape, pos=None, rpy=None, pose=None, friction=None,
                 visible=True, name=None, color='r', link=None):
        X, Y = shape
        super(Contact, self).__init__(
            X, Y, Z=self.THICKNESS, pos=pos, rpy=rpy, pose=pose,
            visible=visible, dZ=-self.THICKNESS, name=name)
        v1 = dot(self.T, array([+self.X, +self.Y, -self.Z, 1.]))[:3]
        v2 = dot(self.T, array([+self.X, -self.Y, -self.Z, 1.]))[:3]
        v3 = dot(self.T, array([-self.X, -self.Y, -self.Z, 1.]))[:3]
        v4 = dot(self.T, array([-self.X, +self.Y, -self.Z, 1.]))[:3]
        v1.flags.writeable = False
        v2.flags.writeable = False
        v3.flags.writeable = False
        v4.flags.writeable = False
        self.friction = friction
        self.link = link
        self.shape = shape
        self.vertices = [v1, v2, v3, v4]

    """
    Geometry
    ========
    """

    def grasp_matrix(self, p):
        """
        Compute the grasp matrix for a given destination point.

        The grasp matrix :math:`G_P` converts the local contact wrench `w` to
        the contact wrench :math:`w_P` at another point `P`:

        .. math::

            w_P = G_P w

        All wrench coordinates being taken in the world frame.

        Parameters
        ----------
        p : array, shape=(3,)
            Point, in world frame coordinates, where the wrench is taken.

        Returns
        -------
        G : ndarray
            Grasp matrix :math:`G_P`.
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

    """
    Force Friction Cone
    ===================

    All linearized friction cones in pymanoid use the inner (conservative)
    approximation. See <https://scaron.info/teaching/friction-model.html>.
    """

    @property
    def force_face(self):
        """
        Face (H-rep) of the force friction cone in world frame.
        """
        mu = self.friction / sqrt(2)  # inner approximation
        face_local = array([
            [-1, 0, -mu],
            [+1, 0, -mu],
            [0, -1, -mu],
            [0, +1, -mu]])
        return dot(face_local, self.R.T)

    @property
    def force_rays(self):
        """
        Rays (V-rep) of the force friction cone in world frame.
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
        Span matrix of the force friction cone in world frame.
        """
        return array(self.force_rays).T

    """
    Wrench Friction Cone
    ====================
    """

    @property
    def wrench_face(self):
        """
        Matrix `F` of friction inequalities in world frame.

        This matrix describes the linearized Coulomb friction model (in the
        fixed contact mode) by:

        .. math::

            F w \leq 0

        where `w` is the contact wrench at the contact point (``self.p``) in the
        world frame. See [CPN15]_ for the derivation of the formula for `F`.

        References
        ----------
        .. [CPN15] Caron, Pham, Nakamura, "Stability of surface contacts for
           humanoid robots: Closed-form formulae of the contact wrench cone for
           rectangular support areas." 2015 IEEE International Conference on
           Robotics and Automation (ICRA).
        """
        X, Y = self.X, self.Y
        mu = self.friction / sqrt(2)  # inner approximation
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
        return [
            hstack([f, cross(v - self.p, f)])
            for v in self.vertices
            for f in self.force_rays]

    @property
    def wrench_span(self):
        """
        Span matrix of the contact wrench cone in world frame.

        This matrix is such that all valid contact wrenches can be written as:

        .. math::

            w_P = S \\lambda, \\quad \\lambda \\geq 0

        where `S` is the friction span and :math:`\\lambda` is a vector with
        positive coordinates.

        Returns
        -------
        S : array, shape=(6, 16)
            Span matrix of the contact wrench cone.

        Notes
        -----
        Note that the contact wrench coordinates :math:`w_P` ("output" of `S`)
        are taken at the contact point `P` (``self.p``) and in the world frame.
        Meanwhile, the number of columns of `S` results from our choice of 4
        contact points (one for each vertex of the rectangular area) with
        4-sided friction pyramids at each.
        """
        return hstack([
            dot(vstack([eye(3), crossmat(v - self.p)]), self.force_span)
            for v in self.vertices])


class ContactSet(object):

    def __init__(self, contacts=None):
        """
        Create a new contact set.

        Parameters
        ----------
        contacts : list of Contact
            List of contacts that define the contact set.
        """
        assert type(contacts) is list
        self.contacts = contacts
        self.nb_contacts = len(self.contacts)

    def compute_grasp_matrix(self, p):
        """
        Compute the grasp matrix of all contact wrenches at point p.

        Parameters
        ----------
        p : array, shape=(3,)
            Point where the resultant wrench is taken at.

        Returns
        -------
        G : array, shape=(6, m)
            Grasp matrix giving the resultant contact wrench :math:`w_P` of all
            contact wrenches as :math:`w_P = G w_{all}`, with :math:`w_{all}`
            the stacked vector of contact wrenches (each wrench being taken at
            its respective contact point and in the world frame).
        """
        return hstack([c.grasp_matrix(p) for c in self.contacts])

    def compute_wrench_face(self, p):
        """
        Compute the face matrix of the contact wrench cone in the world frame.

        Parameters
        ----------
        p : array, shape=(3,)
            Point where the resultant wrench is taken at.

        Returns
        -------
        F : array, shape=(m, 6)
            Friction matrix such that all valid contact wrenches satisfy
            :math:`F w \\leq 0`, where `w` is the resultant contact wrench at
            `p`.
        """
        span_matrix = self.compute_wrench_span(p)
        return compute_cone_face_matrix(span_matrix)

    def compute_wrench_span(self, p):
        """
        Compute the span matrix of the contact wrench cone in world frame.

        Parameters
        ----------
        p : array, shape=(3,)
            Point where the resultant-wrench coordinates are taken.

        Returns
        -------
        S : array, shape=(6, m)
            Span matrix of the net contact wrench cone.

        Notes
        -----
        The span matrix :math:`S_P` such that all valid contact wrenches can be
        written as:

        .. math::

            w_P = S_P \\lambda, \\quad \\lambda \\geq 0

        where :math:`w_P` denotes the contact-wrench coordinates at point `P`.
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

    def find_static_supporting_wrenches(self, com, mass):
        """Find supporting contact wrenches in static-equilibrium.

        Parameters
        ----------
        com : array, shape=(3,)
            Position of the center of mass.
        mass : scalar
            Total mass of the robot in [kg].

        Returns
        -------
        support : list of (Contact, array) couples
            Mapping between each contact `i` in the contact set and a supporting
            contact wrench :math:`w^i_{C_i}`.
        """
        f = array([0., 0., mass * 9.81])
        tau_G = zeros(3)
        wrench = hstack([f, tau_G])
        return self.find_supporting_wrenches(wrench, com)

    def find_supporting_wrenches(self, wrench, point):
        """Find supporting contact wrenches for a given net contact wrench.

        Parameters
        ----------
        wrench : array, shape=(6,)
            Resultant contact wrench :math:`w_P` to be realized.
        point : array, shape=(3,)
            Point `P` where the wrench is expressed.

        Returns
        -------
        support : list of (Contact, array) pairs
            Mapping between each contact `i` in the contact set and a supporting
            contact wrench :math:`w^i_{C_i}`. All contact wrenches satisfy
            friction constraints and sum up to the net wrench: :math:`\\sum_c
            w^i_P = w_P``.

        Note
        ----
        Note that wrench coordinates are returned in their respective contact
        frames (:math:`w^i_{C_i}`), not at the point `P` where the net wrench
        :math:`w_P` is given.
        """
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
        support = [
            (contact, w_all[6 * i:6 * (i + 1)])
            for i, contact in enumerate(self.contacts)]
        return support

    """
    Support areas and volumes
    =========================
    """

    def compute_static_equilibrium_polygon(self, method='hull'):
        """
        Compute the static-equilibrium polygon of the center of mass.

        Parameters
        ----------
        method : string, optional
            choice between 'bretl', 'cdd' or 'hull'

        Returns
        -------
        vertices : list of arrays
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
           August 2008.

        .. [CPN16] Stéphane Caron, Quang-Cuong Pham and Yoshihiko Nakamura, "ZMP
           support areas for multi-contact mobility under frictional
           constraints," IEEE Transactions on Robotics, Dec. 2016.
           `[pdf] <https://scaron.info/papers/journal/caron-tro-2016.pdf>`__

        .. [CK16] Stéphane Caron and Abderrahmane Kheddar, "Multi-contact
           Walking Pattern Generation based on Model Preview Control of 3D COM
           Accelerations," 2016 IEEE-RAS 16th International Conference on
           Humanoid Robots (Humanoids), Cancun, Mexico, 2016, pp. 550-557.
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
        # mass has no effect on the output polygon, see Section IV.B in [CK16]_
        E = 1. / (mass * 9.81) * vstack([-G[4, :], +G[3, :]])
        f = array([p[0], p[1]])
        return project_polytope(
            ineq=(F, zeros(F.shape[0])),
            eq=(G[(0, 1, 2, 5), :], array([0, 0, mass * 9.81, 0])),
            proj=(E, f), method=method)

    def compute_zmp_support_area(self, com, plane, method='bretl'):
        """
        Compute the (pendular) ZMP support area for a given COM position.


        Parameters
        ----------
        com : array, shape=(3,)
            COM position.
        plane : array, shape=(3,)
            Origin of the virtual plane.
        method : string, default='bretl'
            Choice between ``"bretl"`` or ``"cdd"``.

        Returns
        -------
        vertices : list of arrays
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
        B = vstack([
            hstack([z_com * eye(3), crossmat_n]),
            hstack([zeros(3), com])])  # \sim hstack([-(cross(n, p_in)), n])])
        C = 1. / (mass * 9.81) * dot(B, G)
        d = hstack([com, [0]])
        E = (z_zmp - z_com) / (mass * 9.81) * G[:2, :]
        f = array([com[0], com[1]])
        return project_polytope(
            ineq=(F, zeros(F.shape[0])), eq=(C, d), proj=(E, f), method=method)

    def compute_pendular_accel_cone(self, com, zdd_max=None, reduced=False):
        """
        Compute the pendular COM acceleration cone for a given COM position.

        The pendular cone is the reduction of the Contact Wrench Cone when the
        angular momentum at the COM is zero.

        Parameters
        ----------
        com : array, shape=(3,), or list of such arrays
            COM position, or list of COM vertices.
        zdd_max : scalar, optional
            Maximum vertical acceleration in the output cone.
        reduced : bool, optional
            If ``True``, returns the reduced 2D form rather than a 3D cone.

        Returns
        -------
        vertices : list of (3,) arrays
            List of 3D vertices of the (truncated) COM acceleration cone, or of
            the 2D vertices of the reduced form if ``reduced`` is ``True``.

        Notes
        -----
        The method is based on a rewriting of the CWC formula, followed by a 2D
        convex hull on dual vertices. The algorithm is described in [CK16]_.

        When ``com`` is a list of vertices, the returned cone corresponds to COM
        accelerations that are feasible from *all* COM located inside the
        polytope. See [CK16]_ for details on this conservative criterion.
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
        return [gravity] + vertices_at_zdd
