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


from robot_centroid import CentroidalRobot
from contact import ContactSet
from numpy import array, dot, cross, zeros


class Humanoid(CentroidalRobot):

    #
    # Whole-body forces
    #

    def compute_gravito_inertial_wrench(self, qdd, p):
        """
        Compute the gravito-inertial wrench:

            w = [ f   ] = [ m (g - pdd_G)                    ]
                [ tau ]   [ (p_G - p) x m (g - pdd_G) - Ld_G ]

        with m the robot mass, g the gravity vector, G the COM, pdd_G the
        acceleration of the COM, and Ld_GG the rate of change of the angular
        momentum (taken at the COM).

        q -- array of DOF values
        qd -- array of DOF velocities
        qdd -- array of DOF accelerations
        p -- reference point at which the wrench is taken
        """
        g = array([0, 0, -9.81])
        f_gi = self.mass * g
        tau_gi = zeros(3)
        link_velocities = self.rave.GetLinkVelocities()
        link_accelerations = self.rave.GetLinkAccelerations(qdd)
        for link in self.rave.GetLinks():
            mi = link.GetMass()
            ci = link.GetGlobalCOM()
            I_ci = link.GetLocalInertia()
            Ri = link.GetTransform()[0:3, 0:3]
            ri = dot(Ri, link.GetLocalCOM())
            angvel = link_velocities[link.GetIndex()][3:]
            linacc = link_accelerations[link.GetIndex()][:3]
            angacc = link_accelerations[link.GetIndex()][3:]
            ci_ddot = linacc \
                + cross(angvel, cross(angvel, ri)) \
                + cross(angacc, ri)
            angmmt = dot(I_ci, angacc) - cross(dot(I_ci, angvel), angvel)
            f_gi -= mi * ci_ddot[2]
            tau_gi += mi * cross(ci, g - ci_ddot) - dot(Ri, angmmt)
        return f_gi, tau_gi

    def compute_zmp(self, qdd):
        """
        Compute the Zero-tilting Moment Point (ZMP).

        The best introduction I know to this concept is:

            P. Sardain and G. Bessonnet, “Forces acting on a biped robot. center
            of pressure-zero moment point,”
            http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.8014&rep=rep1&type=pdf

        INPUT:

        ``qdd`` -- vector of joint accelerations
        """
        O, n = zeros(3), array([0, 0, 1])
        f_gi, tau_gi = self.compute_gravito_inertial_wrench(qdd, O)
        return cross(n, tau_gi) * 1. / dot(n, f_gi)

    #
    # Posture generation
    #

    def generate_posture_from_contacts(self, contact_set, com_target=None,
                                       *args, **kwargs):
        assert self.ik is not None, \
            "Initialize the IK before generating posture"
        if 'left_foot' in contact_set:
            self.add_contact_task(self.left_foot, contact_set['left_foot'])
        if 'right_foot' in contact_set:
            self.add_contact_task(self.right_foot, contact_set['right_foot'])
        if 'left_hand' in contact_set:
            self.add_contact_task(self.left_hand, contact_set['left_hand'])
        if 'right_hand' in contact_set:
            self.add_contact_task(self.right_hand, contact_set['right_hand'])
        if com_target is not None:
            self.add_com_task(com_target)
        self.add_posture_task(self.q_halfsit)
        # warm start on halfsit posture
        # self.set_dof_values(self.q_halfsit)
        # => no, user may want to override this
        self.solve_ik(*args, **kwargs)

    def generate_posture(self, stance, com_target=None, *args, **kwargs):
        assert self.ik is not None, \
            "Initialize the IK before generating posture"
        if type(stance) is ContactSet:
            return self.generate_posture_from_contacts(stance, com_target)
        if hasattr(stance, 'com'):
            self.add_com_task(stance.com)
        elif hasattr(stance, 'com_target'):
            self.add_com_task(stance.com_target)
        elif com_target is not None:
            self.add_com_task(com_target)
        if hasattr(stance, 'left_foot'):
            self.add_contact_task(
                self.left_foot, stance.left_foot)
        if hasattr(stance, 'right_foot'):
            self.add_contact_task(
                self.right_foot, stance.right_foot)
        if hasattr(stance, 'left_hand'):
            self.add_contact_task(
                self.left_hand, stance.left_hand)
        if hasattr(stance, 'right_hand'):
            self.add_contact_task(
                self.right_hand, stance.right_hand)
        self.add_posture_task(self.q_halfsit)
        # warm start on halfsit posture
        # self.set_dof_values(self.q_halfsit)
        # => no, user may want to override this
        self.solve_ik(*args, **kwargs)
