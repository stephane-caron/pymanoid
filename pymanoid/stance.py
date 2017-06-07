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

from numpy import array, dot

from body import Point
from contact import ContactSet
from misc import norm
from polyhedra import compute_polytope_hrep
from tasks import COMTask, ContactTask, DOFTask, MinVelTask, PostureTask


class Stance(ContactSet):

    """
    A stance is a set of IK tasks.

    Parameters
    ----------
    com : array or Point, optional
        Center-of-mass target.
    left_foot : Contact, optional
        Left-foot contact target.
    right_foot : Contact, optional
        Right-foot contact target.
    left_hand : Contact, optional
        Left-hand contact target.
    right_hand : Contact, optional
        Right-hand contact target.
    """

    def __init__(self, com=None, left_foot=None, right_foot=None,
                 left_hand=None, right_hand=None):
        # NB: do not call the parent (ContactSet) constructor
        if not issubclass(type(com), Point):
            com = Point(com, visible=False)
        self.com = com
        self.dof_tasks = {}
        self.left_foot = left_foot
        self.left_hand = left_hand
        self.right_foot = right_foot
        self.right_hand = right_hand
        self.sep_hrep = None

    def bind(self, robot, reg='posture'):
        tasks = []
        if self.left_foot is not None:
            tasks.append(ContactTask(robot, robot.left_foot, self.left_foot))
        if self.left_hand is not None:
            tasks.append(ContactTask(robot, robot.left_hand, self.left_hand))
        if self.right_foot is not None:
            tasks.append(ContactTask(robot, robot.right_foot, self.right_foot))
        if self.right_hand is not None:
            tasks.append(ContactTask(robot, robot.right_hand, self.right_hand))
        for dof_id, dof_target in self.dof_tasks.iteritems():
            tasks.append(DOFTask(robot, dof_id, dof_target))
        tasks.append(COMTask(robot, self.com))
        if reg == 'posture':
            tasks.append(PostureTask(robot, robot.q_halfsit))
        else:  # default regularization is minimum velocity
            tasks.append(MinVelTask(robot))
        robot.ik.clear_tasks()
        for task in tasks:
            robot.ik.add_task(task)
            print "task", task.name, task.gain, task.weight
        robot.stance = self

    @property
    def bodies(self):
        return filter(None, [
            self.com, self.left_foot, self.left_hand, self.right_foot,
            self.right_hand])

    @property
    def contacts(self):
        return filter(None, [
            self.left_foot, self.left_hand, self.right_foot, self.right_hand])

    @property
    def nb_contacts(self):
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
        for body in self.bodies:
            body.hide()

    def show(self):
        for body in self.bodies:
            body.show()

    def compute_static_equilibrium_polygon(self):
        """
        Compute the contact wrench cone (CWC) and static-equilibrium polygon
        (SEP) of the stance.
        """
        sep_vertices = super(Stance, self).compute_static_equilibrium_polygon()
        self.sep_hrep = compute_polytope_hrep(sep_vertices)
        self.sep_norm = array([norm(a) for a in self.sep_hrep[0]])
        self.sep_vertices = sep_vertices
        return sep_vertices

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
