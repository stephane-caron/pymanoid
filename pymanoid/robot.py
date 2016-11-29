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

from numpy import array, cross, dot, tensordot, zeros
from numpy import concatenate, eye, maximum, minimum, ones, vstack
from os.path import basename, splitext
from warnings import warn

from body import PointMass
from process import Process
from rotations import crossmat, rpy_from_quat
from sim import get_openrave_env


class Robot(object):

    """
    Robot with a fixed base. This class wraps OpenRAVE's Robot type.
    """

    __default_xml = """
    <environment>
        <robot file="%s" name="%s" />
    </environment>
    """

    def __init__(self, path=None, xml=None, qd_lim=None):
        """
        Create a new robot model.

        INPUT:

        - ``path`` -- path to the COLLADA model of the robot
        - ``xml`` -- (optional) string in OpenRAVE XML format
        - ``qd_lim`` -- maximum angular joint velocity (in [rad] / [s])
        """
        assert path is not None or xml is not None
        name = basename(splitext(path)[0])
        if xml is None:
            xml = Robot.__default_xml % (path, name)
        env = get_openrave_env()
        env.LoadData(xml)
        rave = env.GetRobot(name)
        nb_dofs = rave.GetDOF()
        q_min, q_max = rave.GetDOFLimits()
        rave.SetDOFVelocities([0] * nb_dofs)
        rave.SetDOFVelocityLimits([1000.] * nb_dofs)
        if qd_lim is None:
            qd_lim = 10.  # [rad] / [s]; this is already quite fast
        qd_max = +qd_lim * ones(nb_dofs)
        qd_min = -qd_lim * ones(nb_dofs)

        self.has_free_flyer = False
        self.ik = None  # created by self.init_ik()
        self.ik_process = None  # created by self.init_ik()
        self.ik_thread = None
        self.is_visible = True
        self.mass = sum([link.GetMass() for link in rave.GetLinks()])
        self.nb_dofs = nb_dofs
        self.q_max = q_max
        self.q_max.flags.writeable = False
        self.q_min = q_min
        self.q_min.flags.writeable = False
        self.qd_max = qd_max
        self.qd_min = qd_min
        self.qdd_max = None  # set in child class
        self.rave = rave
        self.tau_max = None  # set by hand in child robot class
        self.transparency = 0.  # initially opaque

    """
    Visualization
    =============
    """

    def hide(self):
        self.rave.SetVisible(False)

    def set_color(self, r, g, b):
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetAmbientColor([r, g, b])
                geom.SetDiffuseColor([r, g, b])

    def set_transparency(self, transparency):
        self.transparency = transparency
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    def set_visible(self, visible):
        self.is_visible = visible
        self.rave.SetVisible(visible)

    def show(self):
        self.rave.SetVisible(True)

    """
    Degrees of freedom
    ==================

    OpenRAVE calls "DOF values" what we will also call "joint angles". Same for
    "DOF velocities" and "joint velocities".
    """

    @property
    def q(self):
        return self.rave.GetDOFValues()

    @property
    def qd(self):
        return self.rave.GetDOFVelocities()

    def get_dof_limits(self, dof_indices=None):
        """
        Get the couple (q_min, q_max) of DOF limits.

        INPUT:

        - ``dof_indices`` -- (optional) only compute limits for these indices

        .. NOTE::

            This OpenRAVE function is wrapped because it is too slow in
            practice. On my machine:

                In [1]: %timeit robot.get_dof_limits()
                1000000 loops, best of 3: 205 ns per loop

                In [2]: %timeit robot.rave.GetDOFLimits()
                100000 loops, best of 3: 2.59 µs per loop

            Recall that this function is called at every IK step.
        """
        q_min, q_max = self.q_min, self.q_max
        if dof_indices is not None:
            q_max = q_max[dof_indices]
            q_min = q_min[dof_indices]
        return (q_min, q_max)

    def get_dof_values(self, dof_indices=None):
        if dof_indices is not None:
            return self.rave.GetDOFValues(dof_indices)
        return self.rave.GetDOFValues()

    def get_dof_velocities(self, dof_indices=None):
        if dof_indices is not None:
            return self.rave.GetDOFVelocities(dof_indices)
        return self.rave.GetDOFVelocities()

    def set_dof_limits(self, q_min, q_max, dof_indices=None):
        if self.ik is not None:
            warn("DOF limit updates will not be forwarded to ongoing IK")
        self.rave.SetDOFLimits(q_min, q_max, dof_indices)
        self.q_max.flags.writeable = True
        self.q_min.flags.writeable = True
        if dof_indices is not None:
            assert len(q_min) == len(q_max) == len(dof_indices)
            self.q_max[dof_indices] = q_max
            self.q_min[dof_indices] = q_min
        else:
            assert len(q_min) == len(q_max) == self.nb_dofs
            self.q_max = q_max
            self.q_min = q_min
        self.q_max.flags.writeable = False
        self.q_min.flags.writeable = False

    def set_dof_values(self, q, dof_indices=None, clamp=False):
        if clamp:
            q = minimum(maximum(self.q_min, q), self.q_max)
        if dof_indices is not None:
            return self.rave.SetDOFValues(q, dof_indices)
        return self.rave.SetDOFValues(q)

    def set_dof_velocities(self, qd, dof_indices=None):
        check_dof_limits = 0  # CLA_Nothing
        if dof_indices is not None:
            return self.rave.SetDOFVelocities(qd, check_dof_limits, dof_indices)
        return self.rave.SetDOFVelocities(qd)

    """
    Jacobians and Hessians
    ======================
    """

    def compute_link_jacobian(self, link, p=None):
        """
        Compute the jacobian J(q) of the reference frame of the link, i.e. the
        velocity of the link frame is given by:

            [v omega] = J(q) * qd

        where v and omega are the linear and angular velocities of the link
        frame, respectively.

        INPUT:

        - ``link`` -- link index or pymanoid.Link object
        - ``p`` -- link frame origin (optional: if None, link.p is used)
        - ``q`` -- vector of joint angles (optional: if None, robot.q is used)
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

        - ``link`` -- link index or pymanoid.Link object
        - ``p`` -- point coordinates in world frame (optional, default is the
                   origin of the link reference frame)
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

        - ``link`` -- link index or pymanoid.Link object
        - ``p`` -- link frame origin (optional: if None, link.p is used)
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        H_lin = self.rave.ComputeHessianTranslation(link_index, p)
        H_ang = self.rave.ComputeHessianAxisAngle(link_index)
        H = concatenate([H_lin, H_ang], axis=1)
        return H

    def compute_link_pos_hessian(self, link, p=None):
        """
        Compute the hessian H(q) of a point p on ``link``.

        INPUT:

        - ``link`` -- link index or pymanoid.Link object
        - ``p`` -- point coordinates in world frame (optional, default is the
                   origin of the link reference frame)
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        H = self.rave.ComputeHessianTranslation(link_index, p)
        return H

    """
    Inverse Kinematics
    ==================
    """

    def init_ik(self, active_dofs, doflim_gain=0.5):
        """
        Initialize the IK solver.

        INPUT:

        - ``active_dofs`` -- list of DOFs used by the IK
        - ``doflim_gain`` -- gain between 0 and 1 used for DOF limits
        """
        from ik import VelocitySolver

        class IKProcess(Process):
            def on_tick(_, sim):
                self.step_ik(sim.dt)

        self.ik = VelocitySolver(self, active_dofs, doflim_gain)
        self.ik_process = IKProcess()

    def step_ik(self, dt):
        qd = self.ik.compute_velocity(dt)
        self.set_dof_values(self.q + qd * dt, clamp=True)
        self.set_dof_velocities(qd)

    def solve_ik(self, max_it=1000, conv_tol=1e-5, dt=1e-2, debug=False):
        """
        Compute joint-angles q satisfying all kinematic constraints at best.

        INPUT:

        - ``max_it`` -- maximum number of solver iterations
        - ``conv_tol`` -- stop when cost improvement is less than this threshold
        - ``dt`` -- time step for the differential IK
        - ``debug`` -- print extra debug info

        .. NOTE::

            Good values of dt depend on the weights of the IK tasks. Small
            values make convergence slower, while big values may jeopardize it.
        """
        if debug:
            print "solve_ik(max_it=%d, conv_tol=%e)" % (max_it, conv_tol)
        cost = 100000.
        for itnum in xrange(max_it):
            prev_cost = cost
            cost = self.ik.compute_cost(dt)
            cost_relvar = abs(cost - prev_cost) / prev_cost
            if debug:
                print "%2d: %.3f (%+.2e)" % (itnum, cost, cost_relvar)
            if cost_relvar < conv_tol:
                if abs(cost) > 0.1:
                    warn("IK did not converge to solution. "
                         "Is the problem feasible? "
                         "If so, try restarting from a random guess.")
                break
            self.step_ik(dt)
        return itnum, cost

    """
    Inverse Dynamics
    ================
    """

    def compute_inertia_matrix(self, external_torque=None):
        """
        Compute the inertia matrix of the robot.

        INPUT:

        - ``external_torque`` -- vector of external torques (optional)

        .. NOTE::

            The inertia matrix is the matrix M(q) such that the equations of
            motion are:

                M(q) * qdd + qd.T * C(q) * qd + g(q) = F + external_torque

            with:

            q -- vector of joint angles (DOF values)
            qd -- vector of joint velocities
            qdd -- vector of joint accelerations
            C(q) -- Coriolis tensor (derivative of M(q) w.r.t. q)
            g(q) -- gravity vector
            F -- generalized forces (joint torques, contact wrenches, ...)
            external_torque -- additional torque vector (optional)

            This function applies the unit-vector method described by Walker &
            Orin <https://dx.doi.org/10.1115/1.3139699>. It is inefficient, so
            if you are looking for performance, you should consider more recent
            libraries such as <https://github.com/stack-of-tasks/pinocchio>.
        """
        M = zeros((self.nb_dofs, self.nb_dofs))
        for (i, e_i) in enumerate(eye(self.nb_dofs)):
            tm, _, _ = self.rave.ComputeInverseDynamics(
                e_i, external_torque, returncomponents=True)
            M[:, i] = tm
        return M

    def compute_inverse_dynamics(self, qdd=None, external_torque=None):
        """
        Wrapper around OpenRAVE's ComputeInverseDynamics function, which
        implements the Recursive Newton-Euler algorithm by Walker & Orin
        <https://dx.doi.org/10.1115/1.3139699>.

        The function returns three terms tm, tc and tg such that

            tm = M(q) * qdd
            tc = qd.T * C(q) * qd
            tg = g(q)

        where the equations of motion are written:

            tm + tc + tg = F + external_torque

        INPUT:

        ``qdd`` -- vector of joint accelerations (optional; if not present, the
                   return value for tm will be None)
        ``external_torque`` -- vector of external joint torques (optional)
        """
        if qdd is None:
            _, tc, tg = self.rave.ComputeInverseDynamics(
                zeros(self.nb_dofs), external_torque, returncomponents=True)
            return None, tc, tg
        tm, tc, tg = self.rave.ComputeInverseDynamics(
            qdd, external_torque, returncomponents=True)
        return tm, tc, tg


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
        self.__com_box = None
        self.__comd = None

    def set_free_flyer(self, pos, rpy=None, quat=None):
        if rpy is None and quat is not None:
            rpy = rpy_from_quat(quat)
        self.set_dof_values(pos, [self.TRANS_X, self.TRANS_Y, self.TRANS_Z])
        self.set_dof_values(rpy, [self.ROT_R, self.ROT_P, self.ROT_Y])

    def set_dof_values(self, q, dof_indices=None, clamp=False):
        self.__cam = None
        self.__com = None
        self.__comd = None
        super(Humanoid, self).set_dof_values(
            q, dof_indices=dof_indices, clamp=clamp)

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

    """
    Center Of Mass
    ==============
    """

    @property
    def com(self):
        if self.__com is None:
            self.__com = self.compute_com()
        if self.__com_box is not None:
            self.__com_box.set_pos(self.__com)
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

    def show_com(self):
        if self.__com_box is None:
            self.__com_box = PointMass(self.com, self.mass)
        self.__com_box.show()

    def hide_com(self):
        self.__com_box.hide()

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
        return self.compute_angular_momentum(self.com)

    def compute_cam_jacobian(self):
        """
        Compute the jacobian matrix J(q) such that the CAM is given by:

            L_G(q, qd) = J(q) * qd
        """
        return self.compute_angular_momentum_jacobian(self.com)

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
        return self.compute_angular_momentum_hessian(self.com)

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

        INPUT:

        ``qdd`` -- vector of joint accelerations

        .. NOTE::

            For an excellent introduction to the concepts of ZMP and center of
            pressure, see “Forces acting on a biped robot. center of
            pressure-zero moment point” by P. Sardain and G. Bessonnet
            <http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.138.8014&rep=rep1&type=pdf>.
        """
        O, n = zeros(3), array([0, 0, 1])
        f_gi, tau_gi = self.compute_gravito_inertial_wrench(qdd, O)
        return cross(n, tau_gi) * 1. / dot(n, f_gi)

    """
    Posture generation
    ==================
    """

    def generate_posture(self, stance):
        from tasks import COMTask, ContactTask, PostureTask
        if hasattr(stance, 'com'):
            self.ik.add_task(COMTask(self, stance.com))
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
        self.ik.add_task(PostureTask(self, self.q_halfsit))
        self.solve_ik()
