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

from numpy import array, cross, dot, zeros, tensordot
from os.path import basename, splitext

from contact import ContactSet
from robot import Robot
from rotations import crossmat, rpy_from_quat
from tasks import COMTask, ContactTask, PostureTask, MinAccelerationTask


class Humanoid(Robot):

    """
    Humanoid robots add a free base and centroidal computations.
    """

    __free_flyer_xml = """
    <environment>
        <robot>
            <kinbody>
                <body name="FLYER_TX_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="FLYER_TY_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="FLYER_TZ_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="FLYER_ROLL_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="FLYER_PITCH_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="FLYER_YAW_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <robot file="%s" name="%s">
                <kinbody>
                    <joint name="FLYER_TX" type="slider">
                        <body>FLYER_TX_LINK</body>
                        <body>FLYER_TY_LINK</body>
                        <axis>1 0 0</axis>
                    </joint>
                    <joint name="FLYER_TY" type="slider">
                        <body>FLYER_TY_LINK</body>
                        <body>FLYER_TZ_LINK</body>
                        <axis>0 1 0</axis>
                    </joint>
                    <joint name="FLYER_TZ" type="slider">
                        <body>FLYER_TZ_LINK</body>
                        <body>FLYER_YAW_LINK</body>
                        <axis>0 0 1</axis>
                    </joint>
                    <joint name="FLYER_YAW" type="hinge" circular="true">
                        <body>FLYER_YAW_LINK</body>
                        <body>FLYER_PITCH_LINK</body>
                        <axis>0 0 1</axis>
                    </joint>
                    <joint name="FLYER_PITCH" type="hinge" circular="true">
                        <body>FLYER_PITCH_LINK</body>
                        <body>FLYER_ROLL_LINK</body>
                        <axis>0 1 0</axis>
                    </joint>
                    <joint name="FLYER_ROLL" type="hinge" circular="true">
                        <body>FLYER_ROLL_LINK</body>
                        <body>%s</body>
                        <axis>1 0 0</axis>
                    </joint>
                </kinbody>
            </robot>
        </robot>
    </environment>
    """

    def __init__(self, path, root_body, qd_lim=None):
        """
        Create a new humanoid robot model.

        INPUT:

        - ``path`` -- path to the COLLADA model of the robot
        - ``root_body`` -- name of the root (first) body in the model
        - ``qd_lim`` -- maximum angular joint velocity (in [rad] / [s])
        """
        name = basename(splitext(path)[0])
        xml = Humanoid.__free_flyer_xml % (path, name, root_body)
        super(Humanoid, self).__init__(path, xml=xml, qd_lim=qd_lim)
        self.has_free_flyer = True
        self.__cam = None
        self.__com = None
        self.__comd = None

    def set_free_flyer(self, pos, rpy=None, quat=None):
        if rpy is None and quat is not None:
            rpy = rpy_from_quat(quat)
        self.set_dof_values(pos, [self.TRANS_X, self.TRANS_Y, self.TRANS_Z])
        self.set_dof_values(rpy, [self.ROT_R, self.ROT_P, self.ROT_Y])

    def set_dof_values(self, q, dof_indices=None):
        self.__cam = None
        self.__com = None
        self.__comd = None
        super(Humanoid, self).set_dof_values(q, dof_indices=dof_indices)

    def set_dof_velocities(self, qd, dof_indices=None):
        self.__cam = None
        self.__comd = None
        super(Humanoid, self).set_dof_velocities(qd, dof_indices=dof_indices)

    def set_active_dof_values(self, q_active):
        self.__cam = None
        self.__com = None
        self.__comd = None
        super(Humanoid, self).set_active_dof_values(q_active)

    def set_active_dof_velocities(self, qd_active):
        self.__cam = None
        super(Humanoid, self).set_active_dof_velocities(qd_active)

    def init_ik(self, gains=None, weights=None):
        """
        Initialize the IK solver.

        INPUT:

        - ``gains`` -- dictionary of default task gains
        - ``weights`` -- dictionary of default task weights
        """
        if weights is None:
            weights = {
                'com': 10.,
                'contact': 10000.,
                'link': 100.,
                'posture': 1.,
            }
        super(Humanoid, self).init_ik(gains, weights)

    """
    Center Of Mass
    ==============
    """

    @property
    def com(self):
        if self.__com is None:
            self.__com = self.compute_com()
        return self.__com

    @property
    def comd(self):
        if self.__comd is None:
            self.__comd = self.compute_com_velocity()
        return self.__comd

    def compute_com(self):
        total = zeros(3)
        for link in self.rave.GetLinks():
            m = link.GetMass()
            c = link.GetGlobalCOM()
            total += m * c
        return total / self.mass

    def compute_com_velocity(self):
        total = zeros(3)
        for link in self.rave.GetLinks():
            m = link.GetMass()
            R = link.GetTransform()[0:3, 0:3]
            c_local = link.GetLocalCOM()
            v = link.GetVelocity()
            rd, omega = v[:3], v[3:]
            cd = rd + cross(omega, dot(R, c_local))
            total += m * cd
        return total / self.mass

    def compute_com_jacobian(self):
        """
        Compute the jacobian matrix J(q) of the position of the center of mass G
        of the robot, i.e. the velocity of the G is given by:

                pd_G(q, qd) = J(q) * qd

        q -- vector of joint angles
        qd -- vector of joint velocities
        pd_G -- velocity of the center of mass G
        """
        J_com = zeros((3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            if m < 1e-4:
                continue
            index = link.GetIndex()
            c = link.GetGlobalCOM()
            J_com += m * self.rave.ComputeJacobianTranslation(index, c)
        J_com /= self.mass
        return J_com

    def compute_com_acceleration(self, qdd):
        qd = self.qd
        J = self.compute_com_jacobian()
        H = self.compute_com_hessian()
        return dot(J, qdd) + dot(qd, dot(H, qdd))

    def compute_com_hessian(self):
        H_com = zeros((self.nb_dofs, 3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            if m < 1e-4:
                continue
            index = link.GetIndex()
            c = link.GetGlobalCOM()
            H_com += m * self.rave.ComputeHessianTranslation(index, c)
        H_com /= self.mass
        return H_com

    """
    Angular Momentum
    ================
    """

    def compute_angular_momentum(self, p):
        """
        Compute the angular momentum with respect to point p.

        INPUT:

        - ``p`` -- application point in world coordinates
        """
        am = zeros(3)
        for link in self.rave.GetLinks():
            # TODO: replace by newer link.GetGlobalInertia()
            T = link.GetTransform()
            m = link.GetMass()
            v = link.GetVelocity()
            c = link.GetGlobalCOM()
            R, r = T[0:3, 0:3], T[0:3, 3]
            I = dot(R, dot(link.GetLocalInertia(), R.T))
            rd, omega = v[:3], v[3:]
            cd = rd + cross(r - c, omega)
            am += cross(c - p, m * cd) + dot(I, omega)
        return am

    def compute_angular_momentum_jacobian(self, p):
        """
        Compute the jacobian matrix J(q) such that the angular momentum of the
        robot at p is given by:

            L_p(q, qd) = J(q) * qd

        INPUT:

        - ``p`` -- application point in world coordinates
        """
        J_am = zeros((3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            i = link.GetIndex()
            c = link.GetGlobalCOM()
            R = link.GetTransform()[0:3, 0:3]
            I = dot(R, dot(link.GetLocalInertia(), R.T))
            J_trans = self.rave.ComputeJacobianTranslation(i, c)
            J_rot = self.rave.ComputeJacobianAxisAngle(i)
            J_am += dot(crossmat(c - p), m * J_trans) + dot(I, J_rot)
        return J_am

    def compute_angular_momentum_hessian(self, p):
        """
        Returns a matrix H(q) such that the rate of change of the angular
        momentum with respect to point p is

            Ld_p(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd)),

        where J(q) is the angular-momentum jacobian.

        p -- application point in world coordinates
        """
        def crosstens(M):
            assert M.shape[0] == 3
            Z = zeros(M.shape[1])
            T = array([[Z, -M[2, :], M[1, :]],
                       [M[2, :], Z, -M[0, :]],
                       [-M[1, :], M[0, :], Z]])
            return T.transpose([2, 0, 1])  # T.shape == (M.shape[1], 3, 3)

        def middot(M, T):
            """
            Dot product of a matrix with the mid-coordinate of a 3D tensor.

            M -- matrix with shape (n, m)
            T -- tensor with shape (a, m, b)

            Outputs a tensor of shape (a, n, b).
            """
            return tensordot(M, T, axes=(1, 1)).transpose([1, 0, 2])

        H = zeros((self.nb_dofs, 3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            i = link.GetIndex()
            c = link.GetGlobalCOM()
            R = link.GetTransform()[0:3, 0:3]
            # J_trans = self.rave.ComputeJacobianTranslation(i, c)
            J_rot = self.rave.ComputeJacobianAxisAngle(i)
            H_trans = self.rave.ComputeHessianTranslation(i, c)
            H_rot = self.rave.ComputeHessianAxisAngle(i)
            I = dot(R, dot(link.GetLocalInertia(), R.T))
            H += middot(crossmat(c - p), m * H_trans) \
                + middot(I, H_rot) \
                - dot(crosstens(dot(I, J_rot)), J_rot)
        return H

    """
    Centroidal Angular Momentum (CAM)
    =================================

    It is simply the angular momentum taken at the center of mass.
    """

    @property
    def cam(self):
        if self.__cam is None:
            self.__cam = self.compute_cam()
        return self.__cam

    def compute_cam(self):
        """Compute the centroidal angular momentum."""
        p_G = self.compute_com()
        return self.compute_angular_momentum(p_G)

    def compute_cam_jacobian(self):
        """
        Compute the jacobian matrix J(q) such that the CAM is given by:

            L_G(q, qd) = J(q) * qd
        """
        return self.compute_angular_momentum_jacobian(self.compute_com())

    def compute_cam_rate(self, qdd):
        """Compute the time-derivative of the CAM. """
        qd = self.qd
        J = self.compute_cam_jacobian()
        H = self.compute_cam_hessian()
        return dot(J, qdd) + dot(qd, dot(H, qd))

    def compute_cam_hessian(self, q):
        """
        Compute the matrix H(q) such that the rate of change of the CAM is

            Ld_G(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd))
        """
        p_G = self.compute_com()
        return self.compute_angular_momentum_hessian(p_G)

    """
    Whole-body wrench
    =================
    """

    def compute_gravito_inertial_wrench(self, qdd, p):
        """
        Compute the gravito-inertial wrench:

            w(p) = [ f      ] = [ m (g - pdd_G)                    ]
                   [ tau(p) ]   [ (p_G - p) x m (g - pdd_G) - Ld_G ]

        with m the robot mass, g the gravity vector, G the COM, pdd_G the
        acceleration of the COM, and Ld_GG the rate of change of the angular
        momentum (taken at the COM).

        INPUT:

        - ``qdd`` -- array of DOF accelerations
        - ``p`` -- reference point at which the wrench is taken
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

    """
    Zero-tilting Moment Point
    =========================
    """

    def compute_zmp(self, qdd):
        """
        Compute the Zero-tilting Moment Point (ZMP).

        The best introduction I know to this concept is:

            P. Sardain and G. Bessonnet, “Forces acting on a biped robot. center
            of pressure-zero moment point”
            <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.8014&rep=rep1&type=pdf>.

        INPUT:

        ``qdd`` -- vector of joint accelerations
        """
        O, n = zeros(3), array([0, 0, 1])
        f_gi, tau_gi = self.compute_gravito_inertial_wrench(qdd, O)
        return cross(n, tau_gi) * 1. / dot(n, f_gi)

    """
    Posture generation
    ==================
    """

    def generate_posture_from_contacts(self, contact_set, com_target=None,
                                       regularization='posture', *args,
                                       **kwargs):
        assert self.ik is not None, \
            "Initialize the IK before generating posture"
        assert regularization in ['posture', 'min_acceleration']
        if 'left_foot' in contact_set:
            self.ik.add_task(
                ContactTask(self, self.left_foot, contact_set['left_foot']))
        if 'right_foot' in contact_set:
            self.ik.add_task(
                ContactTask(self, self.right_foot, contact_set['right_foot']))
        if 'left_hand' in contact_set:
            self.ik.add_task(
                ContactTask(self, self.left_hand, contact_set['left_hand']))
        if 'right_hand' in contact_set:
            self.ik.add_task(
                ContactTask(self.right_hand, contact_set['right_hand']))
        if com_target is not None:
            self.ik.add_task(COMTask(self, com_target))
        if regularization == 'posture':
            self.ik.add_task(PostureTask(self, self.q_halfsit))
        else:  # regularization == 'min_acceleration'
            self.ik.add_task(MinAccelerationTask(self))
        self.solve_ik(*args, **kwargs)

    def generate_posture_from_stance(self, stance, com_target=None,
                                     regularization='posture', *args, **kwargs):
        assert regularization in ['posture', 'min_acceleration']
        if hasattr(stance, 'com'):
            self.ik.add_task(COMTask(self, stance.com))
        elif hasattr(stance, 'com_target'):
            self.ik.add_task(COMTask(self, stance.com_target))
        elif com_target is not None:
            self.ik.add_task(COMTask(self, com_target))
        if hasattr(stance, 'left_foot'):
            self.ik.add_task(
                ContactTask(self, self.left_foot, stance.left_foot))
        if hasattr(stance, 'right_foot'):
            self.ik.add_task(
                ContactTask(self, self.right_foot, stance.right_foot))
        if hasattr(stance, 'left_hand'):
            self.ik.add_task(
                ContactTask(self, self.left_hand, stance.left_hand))
        if hasattr(stance, 'right_hand'):
            self.ik.add_task(
                ContactTask(self, self.right_hand, stance.right_hand))
        if regularization == 'posture':
            self.ik.add_task(PostureTask(self, self.q_halfsit))
        else:  # regularization == 'min_acceleration'
            self.ik.add_task(MinAccelerationTask(self))
        self.solve_ik(*args, **kwargs)

    def generate_posture(self, contacts, *args, **kwargs):
        assert self.ik is not None, \
            "Initialize the IK before generating posture"
        if type(contacts) is ContactSet:
            return self.generate_posture_from_contacts(
                contacts, *args, **kwargs)
        else:  # type(contacts) is Stance:
            return self.generate_posture_from_stance(
                contacts, *args, **kwargs)
