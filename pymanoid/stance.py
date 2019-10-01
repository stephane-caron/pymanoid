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

import simplejson

from numpy import array, cross, dot, eye, hstack, vstack, zeros
from scipy.linalg import block_diag
from scipy.spatial.qhull import QhullError
from time import time

from .body import PointMass
from .contact import Contact, ContactSet
from .misc import norm
from .pypoman import compute_polygon_hull, compute_polytope_halfspaces
from .pypoman import project_polytope
from .sim import Process
from .sim import gravity
from .tasks import COMTask, ContactTask, DOFTask, MinVelTask, PostureTask


class Stance(ContactSet):

    """
    Set of contact and center of mass (COM) targets for the humanoid.

    Parameters
    ----------
    com : PointMass
        Center of mass target.
    left_foot : Contact, optional
        Left-foot contact target.
    right_foot : Contact, optional
        Right-foot contact target.
    left_hand : Contact, optional
        Left-hand contact target.
    right_hand : Contact, optional
        Right-hand contact target.
    """

    def __init__(self, com, left_foot=None, right_foot=None,
                 left_hand=None, right_hand=None):
        # NB: do not call the parent (ContactSet) constructor
        assert issubclass(type(com), PointMass), \
            "stance COM should be a PointMass object"
        self.com = com
        self.dof_tasks = {}
        self.left_foot = left_foot
        self.left_hand = left_hand
        self.right_foot = right_foot
        self.right_hand = right_hand
        self.robot = None
        self.sep_hrep = None

    def load(self, path):
        """
        Load stance from JSON file.

        Parameters
        ----------
        path : string
            Path to the JSON file.
        """
        def cfd(contact_dict):
            keys = ['shape', 'friction']
            pose_keys = ['rpy', 'quat', 'pos', 'pose']
            keys.extend([k for k in pose_keys if k in contact_dict])
            return Contact(**{k: contact_dict[k] for k in keys})
        with open(path, 'r') as fp:
            d = simplejson.load(fp)
        if 'com' in d:
            self.com = PointMass(**d['com'])
        self.left_foot = cfd(d['left_foot']) if 'left_foot' in d else None
        self.right_foot = cfd(d['right_foot']) if 'right_foot' in d else None
        self.left_hand = cfd(d['left_hand']) if 'left_hand' in d else None
        self.right_hand = cfd(d['right_hand']) if 'right_hand' in d else None

    def save(self, path):
        """
        Save stance into JSON file.

        Parameters
        ----------
        path : string
            Path to JSON file.
        """
        d = {}
        if self.com is not None:
            d['com'] = {"pos": list(self.com.p)}
        if self.left_foot is not None:
            d['left_foot'] = self.left_foot.dict_repr
        if self.right_foot is not None:
            d['right_foot'] = self.right_foot.dict_repr
        if self.left_hand is not None:
            d['left_hand'] = self.left_hand.dict_repr
        if self.right_hand is not None:
            d['right_hand'] = self.right_hand.dict_repr
        with open(path, 'w') as fp:
            simplejson.dump(d, fp, indent=4, sort_keys=True)

    @staticmethod
    def from_json(path):
        """
        Create a new stance from a JSON file.

        Parameters
        ----------
        path : string
            Path to the JSON file.
        """
        com = PointMass([0., 0., 0.], 0.)
        stance = Stance(com)
        stance.load(path)
        return stance

    def bind(self, robot, reg='posture'):
        """
        Bind stance as robot IK targets.

        Parameters
        ----------
        robot : pymanoid.Robot
            Target robot.
        reg : string, optional
            Regularization task, either "posture" or "min_vel".
        """
        tasks = []
        if self.left_foot is not None:
            self.left_foot.link = robot.left_foot
            tasks.append(ContactTask(robot, robot.left_foot, self.left_foot))
        if self.left_hand is not None:
            self.left_hand.link = robot.left_hand
            tasks.append(ContactTask(robot, robot.left_hand, self.left_hand))
        if self.right_foot is not None:
            self.right_foot.link = robot.right_foot
            tasks.append(ContactTask(robot, robot.right_foot, self.right_foot))
        if self.right_hand is not None:
            self.right_hand.link = robot.right_hand
            tasks.append(ContactTask(robot, robot.right_hand, self.right_hand))
        for dof_id, dof_target in self.dof_tasks.iteritems():
            tasks.append(DOFTask(robot, dof_id, dof_target))
        tasks.append(COMTask(robot, self.com))
        if reg == 'posture':
            tasks.append(PostureTask(robot, robot.q_halfsit))
        else:  # reg == 'min_vel'
            tasks.append(MinVelTask(robot))
        robot.ik.clear()
        for task in tasks:
            robot.ik.add(task)
        robot.stance = self
        robot.wrench_distributor = StanceWrenchDistributor(robot.stance)
        self.robot = robot

    @property
    def bodies(self):
        """
        List of end-effector and COM bodies.
        """
        return filter(None, [
            self.com, self.left_foot, self.left_hand, self.right_foot,
            self.right_hand])

    @property
    def contacts(self):
        """
        List of active contacts.
        """
        return filter(None, [
            self.left_foot, self.left_hand, self.right_foot, self.right_hand])

    @property
    def nb_contacts(self):
        """
        Number of active contacts.
        """
        nb_contacts = 0
        if self.left_foot is not None:
            nb_contacts += 1
        if self.left_hand is not None:
            nb_contacts += 1
        if self.right_foot is not None:
            nb_contacts += 1
        if self.right_hand is not None:
            nb_contacts += 1
        return nb_contacts

    def hide(self):
        """
        Hide end-effector and COM body handles.
        """
        for body in self.bodies:
            body.hide()

    def show(self):
        """
        Show end-effector and COM body handles.
        """
        for body in self.bodies:
            body.show()

    def compute_static_equilibrium_polygon(self, method='hull'):
        """
        Compute the halfspace and vertex representations of the
        static-equilibrium polygon (SEP) of the stance.

        Parameters
        ----------
        method : string, optional
            Which method to use to perform the projection. Choices are 'bretl',
            'cdd' and 'hull' (default).
        """
        sep_vertices = super(Stance, self).compute_static_equilibrium_polygon(
            method=method)
        self.sep_hrep = compute_polytope_halfspaces(sep_vertices)
        self.sep_norm = array([norm(a) for a in self.sep_hrep[0]])
        self.sep_vertices = sep_vertices
        return sep_vertices

    def compute_pendular_accel_cone(self, com_vertices=None, zdd_max=None,
                                    reduced=False):
        """
        Compute the pendular COM acceleration cone of the stance.

        The pendular cone is the reduction of the Contact Wrench Cone when the
        angular momentum at the COM is zero.

        Parameters
        ----------
        com_vertices : list of (3,) arrays, optional
            Vertices of a COM bounding polytope.
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
        convex hull on dual vertices. The algorithm is described in [Caron16]_.

        When ``com`` is a list of vertices, the returned cone corresponds to
        COM accelerations that are feasible from *all* COM located inside the
        polytope. See [Caron16]_ for details on this conservative criterion.
        """
        def expand_reduced_pendular_cone(reduced_hull, zdd_max=None):
            g = -gravity[2]  # gravity constant (positive)
            zdd = +g if zdd_max is None else zdd_max
            vertices_at_zdd = [
                array([a * (g + zdd), b * (g + zdd), zdd])
                for (a, b) in reduced_hull]
            return [gravity] + vertices_at_zdd

        if com_vertices is None:
            com_vertices = [self.com.p]
        CWC_O = self.compute_wrench_inequalities([0., 0., 0.])
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
            return expand_reduced_pendular_cone(reduced_hull, zdd_max)
        except QhullError:
            raise Exception("Cannot compute 2D polar for acceleration cone")

    def compute_zmp_support_area(self, height, method='bretl'):
        """
        Compute an extension of the (pendular) multi-contact ZMP support area
        with optional pressure limits on each contact.

        Parameters
        ----------
        height : array, shape=(3,)
            Height at which the ZMP support area is computed.
        method : string, default='bretl'
            Polytope projection algorithm, can be ``"bretl"`` or ``"cdd"``.

        Returns
        -------
        vertices : list of arrays
            Vertices of the ZMP support area.

        Notes
        -----
        There are two polytope projection algorithms: 'bretl' is adapted from
        in [Bretl08]_ while 'cdd' corresponds to the double-description
        formulation from [Caron17z]_. See the Appendix from [Caron16]_ for a
        performance comparison.
        """
        crossmat_n = array([[0, -1, 0], [1, 0, 0], [0, 0, 0]])  # n = [0, 0, 1]
        G = self.compute_grasp_matrix([0, 0, 0])
        F_list, b_list = zip(*[ct.wrench_hrep for ct in self.contacts])
        F = block_diag(*F_list)
        b = hstack(b_list)
        mass = 42.  # [kg], does not affect the output
        # mass has no effect on the output polygon, c.f. Section IV.C in
        # <https://hal.archives-ouvertes.fr/hal-01349880>
        B1 = hstack([self.com.z * eye(3), crossmat_n])
        B2 = hstack([zeros(3), self.com.p])
        # B2 = hstack([-(cross(n, p_in)), n])]) yields same result
        B = vstack([B1, B2])
        C = 1. / (mass * 9.81) * dot(B, G)
        d = hstack([self.com.p, [0]])
        E = (height - self.com.z) / (mass * 9.81) * G[:2, :]
        f = array([self.com.x, self.com.y])
        return project_polytope(
            proj=(E, f),
            ineq=(F, b),
            eq=(C, d),
            method=method)

    def dist_to_sep_edge(self, com):
        """
        Algebraic distance of a COM position to the edge of the
        static-equilibrium polygon.

        Parameters
        ----------
        com : array, shape=(3,)
            COM position to evaluate the distance from.

        Returns
        -------
        dist : scalar
            Algebraic distance to the edge of the polygon. Inner points get a
            positive value, outer points a negative one.
        """
        A, b = self.sep_hrep
        alg_dists = (b - dot(A, com[:2])) / self.sep_norm
        return min(alg_dists)

    def find_static_supporting_wrenches(self):
        """
        Find supporting contact wrenches in static-equilibrium.

        Returns
        -------
        support : list of (Contact, array) couples
            Mapping between each contact `i` in the contact set and a
            supporting contact wrench :math:`w^i_{C_i}`.
        """
        wrench = hstack([array([0., 0., self.com.mass * 9.81]), zeros(3)])
        return self.find_supporting_wrenches(wrench, self.com.p)

    def free_contact(self, effector_name):
        """
        Free a contact from the stance and get the corresponding end-effector
        target. Its task stays in the IK, but the effector is not considered in
        contact any more (e.g. for contact wrench computations).

        Parameters
        ----------
        effector_name : string
            Name of end-effector target, for example "left_foot".

        Returns
        -------
        effector : pymanoid.Contact
            IK target contact.
        """
        if self.__dict__[effector_name] is None:
            raise Exception("Stance's %s is not in contact" % effector_name)
        effector = self.__dict__[effector_name]
        self.__dict__[effector_name] = None
        return effector

    def set_contact(self, effector):
        """
        Set contact on a effector.

        Parameters
        ----------
        effector : pymanoid.Contact
            IK target contact.
        """
        if effector.link.name == self.robot.left_foot.name:
            effector_name = 'left_foot'
        elif effector.link.name == self.robot.left_hand.name:
            effector_name = 'left_hand'
        elif effector.link.name == self.robot.right_foot.name:
            effector_name = 'right_foot'
        elif effector.link.name == self.robot.right_hand.name:
            effector_name = 'right_hand'
        else:
            raise Exception("Unknown robot effector: " + effector.link.name)
        if self.__dict__[effector_name] is not None:
            raise Exception("Stance's %s already in contact" % effector_name)
        self.__dict__[effector_name] = effector


class StanceWrenchDistributor(Process):

    def __init__(self, stance):
        self.last_bkgnd_switch = None
        self.nb_fails = 0
        self.stance = stance
        super(StanceWrenchDistributor, self).__init__()

    def on_tick(self, sim):
        mass = self.stance.com.mass
        p = self.stance.com.p
        pdd = self.stance.com.pdd
        wrench = hstack([mass * (pdd - sim.gravity), zeros(3)])
        try:
            support = self.stance.find_supporting_wrenches(wrench, p)
            d = {sup[0].name: sup[1] for sup in support}
            for contact in self.stance.contacts:
                contact.link.wrench = d[contact.name]
                contact.wrench = d[contact.name]
        except ValueError:
            self.nb_fails += 1
            sim.viewer.SetBkgndColor([.8, .4, .4])
            self.last_bkgnd_switch = time()
        if self.last_bkgnd_switch is not None \
                and time() - self.last_bkgnd_switch > 0.2:
            sim.viewer.SetBkgndColor([1., 1., 1.])
            self.last_bkgnd_switch = None
