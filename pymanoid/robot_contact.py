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

from contact import Contact
from numpy import array, concatenate, dot, vstack, ndarray
from robot_ik import IKRobot


_oppose_quat = array([-1., -1., -1., -1., +1., +1., +1.])


class ContactingRobot(IKRobot):

    def compute_link_jacobian(self, link, p=None):
        """
        Compute the jacobian J(q) of the reference frame of the link, i.e. the
        velocity of the link frame is given by:

            [v omega] = J(q) * qd

        where v and omega are the linear and angular velocities of the link
        frame, respectively.

        link -- link index or pymanoid.Link object
        p -- link frame origin (optional: if None, link.p is used)
        q -- vector of joint angles (optional: if None, robot.q is used)
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        J_lin = self.rave.ComputeJacobianTranslation(link_index, p)
        J_ang = self.rave.ComputeJacobianAxisAngle(link_index)
        J = vstack([J_lin, J_ang])
        return J

    def compute_link_pose_jacobian(self, link):
        J_trans = self.rave.CalculateJacobian(link.index, link.p)
        or_quat = link.rave.GetTransformPose()[:4]  # don't use link.pose
        J_quat = self.rave.CalculateRotationJacobian(link.index, or_quat)
        if or_quat[0] < 0:  # we enforce positive first coefficients
            J_quat *= -1.
        J = vstack([J_quat, J_trans])
        return J

    def compute_link_pos_jacobian(self, link, p=None):
        """
        Compute the position Jacobian of a point p on a given robot link.

        INPUT:

        ``link`` -- link index or pymanoid.Link object
        ``p`` -- point coordinates in world frame (optional, default is the
                 origin of the link reference frame
        """
        link_index = link if type(link) is int else link.index
        p = link.p if p is None else p
        J = self.rave.ComputeJacobianTranslation(link_index, p)
        return J

    def compute_link_hessian(self, link, p=None):
        """
        Compute the hessian H(q) of the reference frame of the link, i.e. the
        acceleration of the link frame is given by:

            [a omegad] = J(q) * qdd + qd.T * H(q) * qd

        where a and omegad are the linear and angular accelerations of the
        frame, and J(q) is the frame jacobian.

        INPUT:

        ``link`` -- link index or pymanoid.Link object
        ``p`` -- link frame origin (optional: if None, link.p is used)
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        H_lin = self.rave.ComputeHessianTranslation(link_index, p)
        H_ang = self.rave.ComputeHessianAxisAngle(link_index)
        H = concatenate([H_lin, H_ang], axis=1)
        return H

    def add_link_pose_task(self, link, target, gain=None, weight=None,
                           task_type='link'):
        if type(target) is Contact:  # used for ROS communications
            target.robot_link = link.index  # dirty
        if type(target) is list:
            target = array(target)

        def _residual(target_pose):
            residual = target_pose - link.pose
            if dot(residual[0:4], residual[0:4]) > 1.:
                return _oppose_quat * target_pose - link.pose
            return residual

        if hasattr(target, 'effector_pose'):  # needs to come before 'pose'
            def residual(dt):
                return _residual(target.effector_pose)
        elif hasattr(target, 'pose'):
            def residual(dt):
                return _residual(target.pose)
        elif type(target) is ndarray:
            def residual(dt):
                return _residual(target)
        else:  # link frame target should be a pose
            msg = "Target of type %s has no 'pose' attribute" % type(target)
            raise Exception(msg)

        def jacobian():
            return self.compute_link_pose_jacobian(link)

        self.ik.add_task(link.name, residual, jacobian, gain, weight, task_type)

    def add_link_pos_task(self, link, target, gain=None, weight=None):
        if type(target) is list:
            target = array(target)
        if hasattr(target, 'pos'):
            def residual(dt):
                return target.pos - link.p
        elif hasattr(target, 'p'):
            def residual(dt):
                return target.p - link.p
        elif type(target) is ndarray:
            def residual(dt):
                return target - link.p
        else:  # this is an aesthetic comment
            msg = "Target of type %s has no 'pos' attribute" % type(target)
            raise Exception(msg)

        def jacobian():
            return self.compute_link_pos_jacobian(link)

        task_name = 'link-%s' % link.name
        self.ik.add_task(task_name, residual, jacobian, gain, weight)

    def remove_link_task(self, link):
        self.ik.remove_task(link.name)

    """
    Contact
    =======

    A contact task is a link task with higher weight.
    """

    def add_contact_task(self, link, target, gain=None, weight=None):
        return self.add_link_pose_task(
            link, target, gain, weight, task_type='contact')

    def add_contact_vargain_task(self, link, contact, gain, weight, alpha):
        """
        Adds a link task with "variable gain", i.e. where the gain is multiplied
        by a factor alpha() between zero and one. This is a bad solution to
        implement

        INPUT:

        ``link`` -- a pymanoid.Link object
        ``target`` -- a pymanoid.Body, or any object with a ``pose`` field
        ``gain`` -- positional gain between zero and one
        ``weight`` -- multiplier of the task squared residual in cost function
        ``alpha`` -- callable function returning a gain multiplier (float)
        """
        def residual(dt):
            return alpha() * (contact.effector_pose - link.pose)

        def jacobian():
            return self.compute_link_pose_jacobian(link)

        task_name = 'contact-%s' % link.name
        self.ik.add_task(task_name, residual, jacobian, gain, weight)

    def remove_contact_task(self, link):
        self.ik.remove_task(link.name)
