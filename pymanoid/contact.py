#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2018 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import array, cross, diag, dot, eye, hstack, sqrt, vstack, zeros
from scipy.linalg import block_diag

from .body import Box
from .pypoman import compute_cone_face_matrix, compute_polygon_hull
from .pypoman import project_polytope
from .qpsolvers import solve_qp
from .transformations import crossmat


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
        Initial pose. Supersedes ``pos`` and ``rpy`` if they are provided at
        the same time.
    friction : scalar
        Static friction coefficient.
    color : char, optional
        Color code in matplotlib convention ('b' for blue, 'g' for green, ...).
    link : body.Manipulator, optional
        Robot link frame in contact.
    slab_thickness : scalar, optional
        Thickness of the contact slab displayed in the GUI, in [m].
    """

    def __init__(self, shape, pos=None, rpy=None, pose=None, friction=None,
                 color='r', link=None, slab_thickness=0.01):
        X, Y = shape
        super(Contact, self).__init__(
            X, Y, Z=slab_thickness, pos=pos, rpy=rpy, pose=pose, color=color,
            dZ=-slab_thickness)
        inner_friction = None if friction is None else friction / sqrt(2)
        self.friction = friction  # isotropic Coulomb friction
        self.inner_friction = inner_friction  # pyramidal approximation
        self.link = link
        self.shape = shape

    def copy(self, color=None, link=None):
        if color is None:
            color = self.color
        if link is None:
            link = self.link
        return Contact(
            self.shape, pose=self.pose, friction=self.friction, color=color,
            link=link)

    @property
    def dict_repr(self):
        return {
            'friction': self.friction,
            'pos': list(self.p),
            'rpy': list(self.rpy),
            'shape': list(self.shape),
        }

    @property
    def force_inequalities(self):
        """
        Matrix of force friction cone inequalities in the world frame.

        Notes
        -----
        All linearized friction cones in pymanoid use the inner (conservative)
        approximation. See <https://scaron.info/teaching/friction-cones.html>.
        """
        mu = self.inner_friction
        hrep_local = array([
            [-1, 0, -mu],
            [+1, 0, -mu],
            [0, -1, -mu],
            [0, +1, -mu]])
        return dot(hrep_local, self.R.T)

    @property
    def force_rays(self):
        """
        Rays of the force friction cone in the world frame.

        Notes
        -----
        All linearized friction cones in pymanoid use the inner (conservative)
        approximation. See <https://scaron.info/teaching/friction-cones.html>.
        """
        mu = self.inner_friction
        f1 = dot(self.R, [+mu, +mu, +1])
        f2 = dot(self.R, [+mu, -mu, +1])
        f3 = dot(self.R, [-mu, +mu, +1])
        f4 = dot(self.R, [-mu, -mu, +1])
        return [f1, f2, f3, f4]

    @property
    def force_span(self):
        """
        Span matrix of the force friction cone in world frame.

        This matrix `S` is such that all valid contact forces can be written:

        .. math::

            f = S \\lambda, \\quad \\lambda \\geq 0

        Notes
        -----
        All linearized friction cones in pymanoid use the inner (conservative)
        approximation. See <https://scaron.info/teaching/friction-cones.html>.
        """
        return array(self.force_rays).T

    def compute_grasp_matrix(self, p):
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

    @property
    def vertices(self):
        X, Y = self.shape
        v1 = dot(self.T, array([+X, +Y, 0., 1.]))[:3]
        v2 = dot(self.T, array([+X, -Y, 0., 1.]))[:3]
        v3 = dot(self.T, array([-X, -Y, 0., 1.]))[:3]
        v4 = dot(self.T, array([-X, +Y, 0., 1.]))[:3]
        return [v1, v2, v3, v4]

    @property
    def wrench_inequalities(self):
        """
        Matrix `F` of friction inequalities in world frame.

        This matrix describes the linearized Coulomb friction model (in the
        fixed contact mode) by:

        .. math::

            F w \\leq 0

        where `w` is the contact wrench at the contact point (``self.p``) in
        the world frame. See [Caron15]_ for the derivation of the formula for
        `F`.
        """
        X, Y = self.shape
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
        return hstack([c.compute_grasp_matrix(p) for c in self.contacts])

    def compute_static_equilibrium_polygon(self, method='hull'):
        """
        Compute the static-equilibrium polygon of the center of mass.

        Parameters
        ----------
        method : string, optional
            Choice between 'bretl', 'cdd' or 'hull'.

        Returns
        -------
        vertices : list of arrays
            2D vertices of the static-equilibrium polygon.

        Notes
        -----
        The method 'bretl' is adapted from in [Bretl08]_ where the
        static-equilibrium polygon was introduced. The method 'cdd' corresponds
        to the double-description approach described in [Caron17z]_. See the
        Appendix from [Caron16]_ for a performance comparison.
        """
        if method == 'hull':
            A_O = self.compute_wrench_inequalities([0, 0, 0])
            k, a_Oz, a_x, a_y = A_O.shape[0], A_O[:, 2], A_O[:, 3], A_O[:, 4]
            B, c = hstack([-a_y.reshape((k, 1)), +a_x.reshape((k, 1))]), -a_Oz
            return compute_polygon_hull(B, c)
        G_0 = self.compute_grasp_matrix([0., 0., 0.])
        F = block_diag(*[ct.wrench_inequalities for ct in self.contacts])
        mass = 42.  # [kg]
        # mass has no effect on the output polygon, see IV.B in [Caron16]_
        E = 1. / (mass * 9.81) * vstack([-G_0[4, :], +G_0[3, :]])
        f = array([0., 0.])
        return project_polytope(
            proj=(E, f),
            ineq=(F, zeros(F.shape[0])),
            eq=(G_0[(0, 1, 2, 5), :], array([0, 0, mass * 9.81, 0])),
            method=method)

    def compute_wrench_inequalities(self, p):
        """
        Compute the matrix of wrench cone inequalities in the world frame.

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

    def find_supporting_wrenches(self, wrench, point, friction_weight=1e-2,
                                 cop_weight=1., yaw_weight=1e-4,
                                 solver='quadprog'):
        """
        Find supporting contact wrenches for a given net contact wrench.

        Parameters
        ----------
        wrench : array, shape=(6,)
            Resultant contact wrench :math:`w_P` to be realized.
        point : array, shape=(3,)
            Point `P` where the wrench is expressed.
        friction_weight : scalar, optional
            Weight on friction minimization.
        cop_weight : scalar, optional
            Weight on COP deviations from the center of the contact patch.
        solver : string, optional
            Name of the QP solver to use. Default is 'quadprog' (fastest). If
            you are planning on using extremal wrenches, 'cvxopt' is slower but
            works better on corner cases.

        Returns
        -------
        support : list of (Contact, array) pairs
            Mapping between each contact `i` and a supporting contact wrench
            :math:`w^i_{C_i}`. All contact wrenches satisfy friction
            constraints and sum up to the net wrench: :math:`\\sum_c w^i_P =
            w_P``.

        Notes
        -----
        Wrench coordinates are returned in their respective contact frames
        (:math:`w^i_{C_i}`), not at the point `P` where the net wrench
        :math:`w_P` is given.
        """
        n = 6 * self.nb_contacts
        epsilon = min(friction_weight, cop_weight, yaw_weight) * 1e-3
        W_f = diag([friction_weight, friction_weight, epsilon])
        W_tau = diag([cop_weight, cop_weight, yaw_weight])
        P = block_diag(*[
            block_diag(
                dot(ct.R, dot(W_f, ct.R.T)),
                dot(ct.R, dot(W_tau, ct.R.T)))
            for ct in self.contacts])
        q = zeros((n,))
        G = block_diag(*[ct.wrench_inequalities for ct in self.contacts])
        h = zeros((G.shape[0],))  # G * x <= h
        A = self.compute_grasp_matrix(point)
        b = wrench
        w_all = solve_qp(P, q, G, h, A, b, solver=solver)
        if w_all is None:
            return None
        support = [
            (contact, w_all[6 * i:6 * (i + 1)])
            for i, contact in enumerate(self.contacts)]
        return support


class ContactFeed(object):

    def __init__(self, path=None, cyclic=False):
        self.__path = None
        self.contacts = []
        self.cyclic = cyclic
        self.next_contact_id = 0
        #
        if path is not None:
            self.load(path)

    @property
    def first(self):
        return self.contacts[0]

    @property
    def next(self):
        return self.contacts[self.next_contact_id]

    @property
    def last(self):
        return self.contacts[-1]

    def hide(self):
        for contact in self.contacts:
            contact.hide()

    def show(self):
        for contact in self.contacts:
            contact.show()

    def duplicate_last_contact(self, dx=0.05, dy=0.05, dz=0.0):
        contact = self.last.copy()
        contact.set_pos(contact.p + [dx, dy, dz])
        contact.show()
        self.contacts.append(contact)

    def pop(self):
        i = self.next_contact_id
        self.next_contact_id += 1
        if self.next_contact_id >= len(self.contacts):
            if not self.cyclic:
                return None
            self.next_contact_id = 0
        return self.contacts[i]

    def load(self, path):
        import simplejson
        assert path.endswith('.json')
        with open(path, 'r') as fp:
            contact_defs = simplejson.load(fp)
        for d in contact_defs:
            self.contacts.append(Contact(
                shape=d['shape'],
                pos=d['pos'],
                rpy=d['rpy'],
                friction=d['friction']))
        self.__path = path

    def save(self, path=None):
        import simplejson
        if path is None:
            path = self.__path
        assert path.endswith('.json')
        contact_defs = [{
            'shape': contact.shape,
            'pos': list(contact.p),
            'rpy': list(contact.rpy),
            'friction': contact.friction}
            for contact in self.contacts]
        with open(path, 'w') as fp:
            simplejson.dump(contact_defs, fp, indent=4, sort_keys=True)
