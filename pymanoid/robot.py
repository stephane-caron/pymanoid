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

from numpy import array, cross, dot, eye, hstack, vstack, zeros
from numpy import concatenate, maximum, minimum
from os.path import basename, isfile, splitext
from warnings import warn

from .body import PointMass
from .gui import draw_line, draw_point
from .ik import IKSolver
from .misc import matplotlib_to_rgb, middot
from .sim import get_openrave_env, gravity
from .transformations import crossmat, quat_from_rpy, rpy_from_quat
from .transformations import rotation_matrix_from_rpy


class Robot(object):

    """
    Robot with a fixed base. This class wraps OpenRAVE's Robot type.

    Parameters
    ----------
    path : string
        Path to the COLLADA model of the robot.
    xml : string, optional
        Environment description in `OpenRAVE XML format
        <http://openrave.programmingvision.com/wiki/index.php/Format:XML>`_.
    """

    __default_xml = """
    <environment>
        <robot file="%s" name="%s" />
    </environment>
    """

    def __init__(self, path=None, xml=None):
        assert path is not None or xml is not None
        name = basename(splitext(path)[0])
        if xml is None:
            xml = Robot.__default_xml % (path, name)
        env = get_openrave_env()
        env.LoadData(xml)
        rave = env.GetRobot(name)
        nb_dofs = rave.GetDOF()
        q_min, q_max = rave.GetDOFLimits()
        rave.SetDOFVelocityLimits([1000.] * nb_dofs)
        self.has_free_flyer = False
        self.ik = None
        self.is_visible = True
        self.mass = sum([link.GetMass() for link in rave.GetLinks()])
        self.nb_dofs = nb_dofs
        self.q_max = q_max
        self.q_max.flags.writeable = False
        self.q_min = q_min
        self.q_min.flags.writeable = False
        self.qd_lim = rave.GetDOFVelocityLimits()
        self.rave = rave
        self.stance = None
        self.tau_max = None  # set by hand in child robot class
        self.transparency = 0.  # initially opaque
        #
        self.ik = IKSolver(self)

    """
    Visualization
    =============
    """

    def hide(self):
        """Make the robot invisible."""
        self.is_visible = False
        self.rave.SetVisible(False)

    def set_color(self, color):
        """
        Set the color of the robot.

        Parameters
        ----------
        color : tuple or string
            RGB tuple, or color code in matplotlib convention.
        """
        if type(color) is str:
            color = matplotlib_to_rgb(color)
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetAmbientColor(color)
                geom.SetDiffuseColor(color)

    def set_transparency(self, transparency):
        """
        Update robot transparency.

        Parameters
        ----------
        transparency : scalar
            Transparency value from 0 (opaque) to 1 (invisible).
        """
        self.transparency = transparency
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    def show(self):
        """Make the robot visible."""
        self.is_visible = True
        self.rave.SetVisible(True)

    """
    Degrees of freedom
    ==================

    OpenRAVE calls "DOF values" what we will also call "joint angles". Same for
    "DOF velocities" and "joint velocities".
    """

    @property
    def q(self):
        """Vector of DOF values."""
        return self.rave.GetDOFValues()

    @property
    def qd(self):
        """Vector of DOF velocities."""
        return self.rave.GetDOFVelocities()

    def get_dof_limits(self, dof_indices=None):
        """
        Get the pair :math:`(q_\\mathrm{min}, q_\\mathrm{max})` of DOF limits.

        Parameters
        ----------
        dof_indices : list of DOF indexes, optional
            Only compute limits for these indices.

        Returns
        -------
        q_min : array
            Vector of minimal DOF values.
        q_max : array
            Vector of maximal DOF values.

        Notes
        -----
        This OpenRAVE function is wrapped because it is too slow in practice.
        On my machine:

            In [1]: %timeit robot.get_dof_limits()
            1000000 loops, best of 3: 237 ns per loop

            In [2]: %timeit robot.rave.GetDOFLimits()
            100000 loops, best of 3: 9.24 µs per loop
        """
        q_min, q_max = self.q_min, self.q_max
        if dof_indices is not None:
            q_max = q_max[dof_indices]
            q_min = q_min[dof_indices]
        return (q_min, q_max)

    def get_dof_values(self, dof_indices=None):
        """
        Get DOF values for a set of DOF indices.

        Parameters
        ----------
        dof_indices : list of integers
            List of DOF indices.

        Returns
        -------
        q_indices : array
            Vector of DOF values for these indices.
        """
        if dof_indices is not None:
            return self.rave.GetDOFValues(dof_indices)
        return self.rave.GetDOFValues()

    def get_dof_velocities(self, dof_indices=None):
        """
        Get DOF velocities for a set of DOF indices.

        Parameters
        ----------
        dof_indices : list of integers
            List of DOF indices.

        Returns
        -------
        qd_indices : array
            Vector of DOF velocities for these indices.
        """
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
        """
        Set the joint values of the robot.

        Parameters
        ----------
        q : list or array
           Joint angle values, ordered by DOF indices.
        dof_indices : list, optional
            List of DOF indices to update.
        clamp : bool
            Correct joint angles when exceeding joint limits.
        """
        if clamp:
            q = minimum(maximum(self.q_min, q), self.q_max)
        if dof_indices is not None:
            return self.rave.SetDOFValues(q, dof_indices)
        return self.rave.SetDOFValues(q)

    def set_dof_velocities(self, qd, dof_indices=None):
        """
        Set the joint velocities of the robot.

        Parameters
        ----------
        qd : list or array
           Joint angle velocities, ordered by DOF indices.
        dof_indices : list, optional
            List of DOF indices to update.
        """
        check_dof_limits = 0  # CLA_Nothing
        if dof_indices is not None:
            return self.rave.SetDOFVelocities(
                qd, check_dof_limits, dof_indices)
        return self.rave.SetDOFVelocities(qd)

    """
    Jacobians and Hessians
    ======================
    """

    def compute_link_jacobian(self, link, p=None):
        """
        Compute the Jacobian `J(q)` of a frame attached to a given link, the
        velocity of this frame being given by:

        .. math::

            \\left[\\begin{array}{c} v_p \\\\ \\omega \\end{array}\\right]
            = J(q) \\dot{q}

        where :math:`v_p` is the linear velocity of the link at point `p`
        (default is the origin of the link frame) and :math:`\\omega` is the
        angular velocity of the link.

        Parameters
        ----------
        link : integer or pymanoid.Link
            Link identifier: either a link index, or the Link object directly.
        p : array
            Point coordinates in the world frame.
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        J_lin = self.rave.ComputeJacobianTranslation(link_index, p)
        J_ang = self.rave.ComputeJacobianAxisAngle(link_index)
        J = vstack([J_lin, J_ang])
        return J

    def compute_link_pose_jacobian(self, link):
        """
        Compute the pose Jacobian of a given link, i.e. the matrix `J(q)` such
        that:

        .. math::

            \\left[\\begin{array}{c} \\dot{\\xi} \\\\ v_L \\end{array}\\right]
            = J(q) \\dot{q},

        with :math:`\\xi` a quaternion for the link orientation and :math:`v_L
        = \\dot{p}_L` the velocity of the origin `L` of the link frame, so that
        the link pose is :math:`[\\xi p_L]` and the left-hand side of the
        equation above is its time-derivative.

        Parameters
        ----------
        link : integer or pymanoid.Link
            Link identifier: either a link index, or the Link object directly.
        """
        J_trans = self.rave.CalculateJacobian(link.index, link.p)
        or_quat = link.rave.GetTransformPose()[:4]  # don't use link.pose
        J_quat = self.rave.CalculateRotationJacobian(link.index, or_quat)
        if or_quat[0] < 0:  # we enforce positive first coefficients
            J_quat *= -1.
        J = vstack([J_quat, J_trans])
        return J

    def compute_link_pos_jacobian(self, link, p=None):
        """
        Compute the position Jacobian of a point `p` on a given robot link.

        Parameters
        ----------
        link : integer or pymanoid.Link
            Link identifier: either a link index, or the Link object directly.
        p : array
            Point coordinates in the world frame.
        """
        link_index = link if type(link) is int else link.index
        p = link.p if p is None else p
        J = self.rave.ComputeJacobianTranslation(link_index, p)
        return J

    def compute_link_hessian(self, link, p=None):
        """
        Compute the Hessian `H(q)` of a frame attached to a robot link, the
        acceleration of which is given by:

        .. math::

            \\left[\\begin{array}{c} a_p \\\\ \\dot{\\omega}
            \\end{array}\\right] = J(q) \\ddot{q} + \\dot{q}^T H(q) \\dot{q}

        where :math:`a_p` is the linear acceleration of the point `p` (default
        is the origin of the link frame) and :math:`\\dot{\\omega}` is the
        angular accelerations of the frame.

        Parameters
        ----------
        link : integer or pymanoid.Link
            Link identifier: either a link index, or the Link object directly.
        p : array
            Point coordinates in the world frame.
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        H_lin = self.rave.ComputeHessianTranslation(link_index, p)
        H_ang = self.rave.ComputeHessianAxisAngle(link_index)
        H = concatenate([H_lin, H_ang], axis=1)
        return H

    def compute_link_pos_hessian(self, link, p=None):
        """
        Compute the translation Hessian H(q) of a point `p` on ``link``, i.e.
        the matrix such that the acceleration of `p` is given by:

        .. math::

            a_p = J(q) \\ddot{q} + \\dot{q}^T H(q) \\dot{q}.

        Parameters
        ----------
        link : integer or pymanoid.Link
            Link identifier: either a link index, or the Link object directly.
        p : array
            Point coordinates in the world frame.
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        H = self.rave.ComputeHessianTranslation(link_index, p)
        return H

    """
    Inverse Dynamics
    ================
    """

    def compute_inertia_matrix(self):
        """
        Compute the inertia matrix :math:`M(q)` of the robot, that is, the
        matrix such that the equations of motion can be written:

        .. math::

            M(q) \\ddot{q} + \\dot{q}^T C(q) \\dot{q} + g(q) = S^T \\tau +
            J_c(q)^T F

        with:

        - :math:`q` -- vector of joint angles (DOF values)
        - :math:`\\dot{q}` -- vector of joint velocities
        - :math:`\\ddot{q}` -- vector of joint accelerations
        - :math:`C(q)` -- Coriolis tensor (derivative of M(q) w.r.t. q)
        - :math:`g(q)` -- gravity vector
        - :math:`S` -- selection matrix on actuated joints
        - :math:`\\tau` -- vector of actuated joint torques
        - :math:`J_c(q)` -- matrix of stacked contact Jacobians
        - :math:`F` -- vector of stacked contact wrenches

        Returns
        -------
        M : array
            Inertia matrix for the robot's current joint angles.

        Notes
        -----
        This function applies the unit-vector method described by Walker & Orin
        in [Walker82]_. It is not efficient, so if you are looking for
        performance, you should consider more recent libraries such as
        `pinocchio <https://github.com/stack-of-tasks/pinocchio>`_ or `RBDyn
        <https://github.com/jrl-umi3218/RBDyn>`_.
        """
        M = zeros((self.nb_dofs, self.nb_dofs))
        for (i, e_i) in enumerate(eye(self.nb_dofs)):
            tm, _, _ = self.rave.ComputeInverseDynamics(
                e_i, None, returncomponents=True)
            M[:, i] = tm
        return M

    def compute_inverse_dynamics(self, qdd=None, external_torque=None):
        """
        Wrapper around OpenRAVE's inverse dynamics function, which implements
        the Recursive Newton-Euler algorithm by Walker & Orin [Walker82]_. It
        returns the three terms :math:`t_m`, :math:`t_c` and :math:`t_g` such
        that:

        .. math::

            t_m & = M(q) \\ddot{q} \\\\
            t_c & = \\dot{q}^T C(q) \\dot{q} \\\\
            t_g & = g(q)

        where the equations of motion are written:

        .. math::

            t_m + t_c + t_g = S^T \\tau + J_c(q)^T F

        Parameters
        ----------
        qdd : array, optional
            Vector :math:`\\ddot{q}` of joint accelerations.
        external_torque : array, optional
            Additional vector of external joint torques for the right-hand side
            of the equations of motion.

        Returns
        -------
        tm : array
            Torques due to inertial accelerations. Will be ``None`` if ``qdd``
            is not provided.
        tc : array
            Torques due to nonlinear (centrifugal and Coriolis) effects.
        tg : array
            Torques due to gravity.
        """
        if qdd is None:
            _, tc, tg = self.rave.ComputeInverseDynamics(
                zeros(self.nb_dofs), external_torque, returncomponents=True)
            return None, tc, tg
        tm, tc, tg = self.rave.ComputeInverseDynamics(
            qdd, external_torque, returncomponents=True)
        return tm, tc, tg

    def compute_static_torques(self, external_torque=None):
        """
        Compute static-equilibrium torques for the manipulator.

        Parameters
        ----------
        external_torque : array
            Vector of external joint torques.

        Returns
        -------
        tg : array
            Vector of static joint torques compensating gravity, and
            ``external_torque`` if applicable.
        """
        qd = self.qd
        qz = zeros(self.nb_dofs)
        self.set_dof_velocities(qz)
        tg = self.rave.ComputeInverseDynamics(qz, external_torque)
        self.set_dof_velocities(qd)
        return tg


class Humanoid(Robot):

    """
    Humanoid robots add a free base and centroidal computations.
    """

    __free_flyer_xml = """
    <environment>
        <robot>
            <kinbody>
                <body name="TRANS_X_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="TRANS_Y_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="TRANS_Z_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="ROT_Y_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="ROT_P_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <kinbody>
                <body name="ROT_R_LINK">
                    <mass type="mimicgeom">
                        <total>0</total>
                    </mass>
                </body>
            </kinbody>
            <robot file="%s" name="%s">
                <kinbody>
                    <joint name="TRANS_X" type="slider">
                        <body>TRANS_X_LINK</body>
                        <body>TRANS_Y_LINK</body>
                        <axis>1 0 0</axis>
                    </joint>
                    <joint name="TRANS_Y" type="slider">
                        <body>TRANS_Y_LINK</body>
                        <body>TRANS_Z_LINK</body>
                        <axis>0 1 0</axis>
                    </joint>
                    <joint name="TRANS_Z" type="slider">
                        <body>TRANS_Z_LINK</body>
                        <body>ROT_Y_LINK</body>
                        <axis>0 0 1</axis>
                    </joint>
                    <joint name="ROT_Y" type="hinge" circular="true">
                        <body>ROT_Y_LINK</body>
                        <body>ROT_P_LINK</body>
                        <axis>0 0 1</axis>
                    </joint>
                    <joint name="ROT_P" type="hinge" circular="true">
                        <body>ROT_P_LINK</body>
                        <body>ROT_R_LINK</body>
                        <axis>0 1 0</axis>
                    </joint>
                    <joint name="ROT_R" type="hinge" circular="true">
                        <body>ROT_R_LINK</body>
                        <body>%s</body>
                        <axis>1 0 0</axis>
                    </joint>
                </kinbody>
            </robot>
        </robot>
    </environment>
    """

    def __init__(self, path, root_body):
        """
        Create a new humanoid robot model.

        Parameters
        ----------
        path : string
            Path to the COLLADA model of the robot.
        root_body : string
            Name of the root body in the kinematic chain.
        """
        if not isfile(path):
            raise Exception("Model file '%s' not found" % path)
        name = basename(splitext(path)[0])
        xml = Humanoid.__free_flyer_xml % (path, name, root_body)
        super(Humanoid, self).__init__(path, xml=xml)
        self.has_free_flyer = True
        self.__cam = None
        self.__com = None
        self.__com_handle = None
        self.__comd = None
        self.__show_com = False

    """
    Kinematics
    ==========
    """

    @property
    def b(self):
        """Binormal vector directing the `y`-axis of the free-flyer frame."""
        return self.R[:, 1]

    def get_dof_name_from_index(self, index):
        """
        Get DOF name from its index in the kinematic chain.

        Parameters
        ----------
        index : integer
            DOF index.

        Returns
        -------
        name : string
            DOF name.
        """
        return str(self.rave.GetJointFromDOFIndex(index).GetName())

    @property
    def n(self):
        """Normal vector directing the `z`-axis of the free-flyer frame."""
        return self.R[:, 2]

    @property
    def p(self):
        """
        Position of the free-flyer in the world frame.

        Returns
        -------
        p : array, shape=(3,)
            Position coordinates in the world frame.
        """
        return self.get_dof_values([self.TRANS_X, self.TRANS_Y, self.TRANS_Z])

    @property
    def rpy(self):
        """
        Orientation of the free-flyer in the world frame.

        Returns
        -------
        rpy : array, shape=(3,)
            Roll-pitch-yaw angles, corresponding to Euler sequence (1, 2, 3).
        """
        return self.get_dof_values([self.ROT_R, self.ROT_P, self.ROT_Y])

    @property
    def quat(self):
        """
        Quaternion of the free-flyer orientation in the world frame.

        Returns
        -------
        quat : array, shape=(4,)
            Quaternion vector (`w`, `x`, `y`, `z`).
        """
        return quat_from_rpy(self.rpy)

    @property
    def pose(self):
        """
        Pose of the free-flyer in the world frame.

        Returns
        -------
        pose : array, shape=(7,)
            Pose in OpenRAVE format (`qw`, `qx`, `qy`, `qz`, `x`, `y`, `z`).
        """
        return hstack([self.quat, self.p])

    @property
    def R(self):
        """
        Rotation matrix `R` of from free-flyer to the world frame.

        Returns
        -------
        R : array, shape=(3, 3)
            Rotation matrix.
        """
        return rotation_matrix_from_rpy(self.rpy)

    @property
    def T(self):
        """
        Rotation matrix `R` of from free-flyer to the world frame.

        Returns
        -------
        R : array, shape=(3, 3)
            Rotation matrix.
        """
        T = zeros((4, 4))
        T[:3, :3] = self.R
        T[:3, 3] = self.p
        T[3, 3] = 1.
        return T

    def set_dof_values(self, q, dof_indices=None, clamp=False):
        """
        Set the joint values of the robot.

        Parameters
        ----------
        q : array
            Vector of joint angle values (ordered by DOF indices).
        dof_indices : list of integers, optional
            List of DOF indices to update.
        clamp : bool, optional
            Correct `q` if it exceeds joint limits (not done by default).
        """
        self.__cam = None
        self.__com = None
        self.__comd = None
        super(Humanoid, self).set_dof_values(
            q, dof_indices=dof_indices, clamp=clamp)
        if self.__show_com:
            self.show_com()

    def set_dof_velocities(self, qd, dof_indices=None):
        """
        Set the joint velocities of the robot.

        Parameters
        ----------
        q : array
            Vector of joint angular velocities (ordered by DOF indices).
        dof_indices : list of integers, optional
            List of DOF indices to update.
        """
        self.__cam = None
        self.__comd = None
        super(Humanoid, self).set_dof_velocities(qd, dof_indices=dof_indices)

    def set_pos(self, pos):
        """
        Set the position of the free-flyer, a.k.a. free-floating or base frame
        of the robot.

        Parameters
        ----------
        pos : array, shape=(3,)
            Position coordinates in the world frame.
        """
        self.set_dof_values(pos, [self.TRANS_X, self.TRANS_Y, self.TRANS_Z])

    def set_rpy(self, rpy):
        """
        Set the orientation of the free-flyer, a.k.a. free-floating or base
        frame of the robot.

        Parameters
        ----------
        rpy : array, shape=(3,)
            Roll-pitch-yaw angles, corresponding to Euler sequence (1, 2, 3).
        """
        self.set_dof_values(rpy, [self.ROT_R, self.ROT_P, self.ROT_Y])

    def set_quat(self, quat):
        """
        Set the orientation of the free-flyer, a.k.a. free-floating or base
        frame of the robot.

        Parameters
        ----------
        quat : array, shape=(4,)
            Quaternion vector (`w`, `x`, `y`, `z`).
        """
        self.set_rpy(rpy_from_quat(quat))

    def set_pose(self, pose):
        """
        Set the pose of the free-flyer, a.k.a. free-floating or base frame of
        the robot.

        Parameters
        ----------
        pose : array, shape=(7,)
            Pose in OpenRAVE format (`qw`, `qx`, `qy`, `qz`, `x`, `y`, `z`).
        """
        self.set_quat(pose[:4])
        self.set_pos(pose[4:])

    def set_x(self, x):
        self.set_dof_values([x], [self.TRANS_X])

    def set_y(self, y):
        self.set_dof_values([y], [self.TRANS_Y])

    def set_z(self, z):
        self.set_dof_values([z], [self.TRANS_Z])

    @property
    def t(self):
        """Tangent vector directing the `x`-axis of the free-flyer frame."""
        return self.R[:, 0]

    """
    Center Of Mass
    ==============
    """

    @property
    def com(self):
        """Position of the center of mass in the world frame."""
        if self.__com is None:
            self.__com = self.compute_com()
        return self.__com

    @property
    def comd(self):
        """Velocity of the center of mass in the world frame."""
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
        Compute the Jacobian matrix :math:`J_\\mathrm{com}(q)` of the center of
        mass `G`, such that the velocity of `G` in the world frame is:

        .. math::

                \\dot{p}_G = J_\\mathrm{com}(q) \\dot{q}

        Returns
        -------
        J_com : array
            Jacobian of the center of mass.
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
        """
        Compute COM acceleration from joint accelerations :math:`\\ddot{q}`.

        Parameters
        ----------
        qdd : array
            Vector of DOF accelerations.

        Returns
        -------
        comdd : (3,) array
            COM acceleration.
        """
        Pd = zeros(3)  # rate-of-change of the linear momentum
        link_velocities = self.rave.GetLinkVelocities()
        link_accelerations = self.rave.GetLinkAccelerations(qdd)
        for link in self.rave.GetLinks():
            R = link.GetTransform()[0:3, 0:3]
            i = link.GetIndex()
            m = link.GetMass()
            omega = link_velocities[i][3:]
            omegad = link_accelerations[i][3:]
            pdd = link_accelerations[i][:3]
            r = dot(R, link.GetLocalCOM())
            Pd += m * (pdd + cross(omega, cross(omega, r)) + cross(omegad, r))
        return Pd / self.mass

    def compute_com_hessian(self):
        """
        Compute the Hessian matrix `H(q)` of the center of mass `G`, such that
        the acceleration of `G` in the world frame is:

        .. math::

                \\ddot{p}_G = J(q) \\ddot{q} + \\dot{q}^T H(q) \\dot{q}
        """
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

    def get_com_point_mass(self):
        """
        Get the center of mass as a PointMass instance.

        Returns
        -------
        com : PointMass
            Center of mass of the robot.
        """
        return PointMass(pos=self.com, mass=self.mass)

    def show_com(self):
        """Show a red ball at the location of the center of mass."""
        self.__show_com = True
        self.__com_handle = [
            draw_point(self.com, pointsize=0.0005 * self.mass, color='r'),
            draw_line(
                self.com, self.com + [0., 0., -1.], linewidth=2., color='y')]

    def hide_com(self):
        """Hide center of mass."""
        self.__show_com = False
        self.__com_handle = None

    """
    Angular Momentum
    ================
    """

    def compute_angular_momentum(self, p):
        """
        Compute the angular momentum with respect to point `p`.

        Parameters
        ----------
        p : array, shape=(3,)
            Application point `p` in world coordinates.

        Returns
        -------
        am : array, shape=(3,)
            Angular momentum of the robot at `p`.
        """
        am = zeros(3)
        for link in self.rave.GetLinks():
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
        Compute the Jacobian matrix J(q) such that the angular momentum of the
        robot at `P` is given by:

        .. math::

            L_P(q, \\dot{q}) = J(q) \\dot{q}

        Parameters
        ----------
        p : array, shape=(3,)
            Application point `P` in world coordinates.

        Returns
        -------
        J_am : array, shape=(3,)
            Jacobian giving the angular momentum of the robot at `P`.
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
        Returns the Hessian tensor :math:`H(q)` such that the rate of change of
        the angular momentum with respect to point `P` is

        .. math::

            \\dot{L}_P(q, \\dot{q}) = J(q) \\ddot{q} + \\dot{q}^T H(q) \\dot{q}

        where :math:`J(q)` is the angular-momentum jacobian.

        Parameters
        ----------
        p : array, shape=(3,)
            Application point in world coordinates.

        Returns
        -------
        H : array, shape=(3, 3, 3)
            Hessian tensor of the angular momentum at the application point.
        """
        def crosstens(M):
            assert M.shape[0] == 3
            Z = zeros(M.shape[1])
            T = array([[Z, -M[2, :], M[1, :]],
                       [M[2, :], Z, -M[0, :]],
                       [-M[1, :], M[0, :], Z]])
            return T.transpose([2, 0, 1])  # T.shape == (M.shape[1], 3, 3)

        H = zeros((self.nb_dofs, 3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            i = link.GetIndex()
            c = link.GetGlobalCOM()
            R = link.GetTransform()[0:3, 0:3]
            J_rot = self.rave.ComputeJacobianAxisAngle(i)
            H_trans = self.rave.ComputeHessianTranslation(i, c)
            H_rot = self.rave.ComputeHessianAxisAngle(i)
            I = dot(R, dot(link.GetLocalInertia(), R.T))
            H += middot(crossmat(c - p), m * H_trans) \
                + middot(I, H_rot) \
                - dot(crosstens(dot(I, J_rot)), J_rot)
        return H

    """
    Centroidal Angular Momentum
    ===========================
    """

    @property
    def cam(self):
        """Centroidal angular momentum."""
        if self.__cam is None:
            self.__cam = self.compute_cam()
        return self.__cam

    def compute_cam(self):
        """Compute the centroidal angular momentum."""
        return self.compute_angular_momentum(self.com)

    def compute_cam_jacobian(self):
        """
        Compute the jacobian matrix `J(q)` such that the CAM is given by:

        .. math::

            L_G(q, \\dot{q}) = J(q) \\dot{q}

        Returns
        -------
        J_cam : array
            Jacobian matrix mapping to the centroidal angular momentum.
        """
        return self.compute_angular_momentum_jacobian(self.com)

    def compute_cam_rate(self, qdd):
        """
        Compute the time-derivative of the CAM.

        Parameters
        ----------
        qdd : array
            Vector of joint accelerations.

        Returns
        -------
        cam : array
            Coordinates of the angular momentum at the COM.

        Note
        ----
        This function is not optimized.
        """
        J = self.compute_cam_jacobian()
        H = self.compute_cam_hessian()
        return dot(J, qdd) + dot(self.qd, dot(H, self.qd))

    def compute_cam_hessian(self, q):
        """
        Compute the matrix H(q) such that the rate of change of the CAM is

        .. math::

            \\dot{L}_G(q, \\dot{q}) = J(q) \\ddot{q} + \\dot{q}^T H(q) \\dot{q}

        Parameters
        ----------
        q : array
            Vector of joint angles.

        Returns
        -------
        H : array
            Hessian matrix for the centroidal angular-momentum.
        """
        return self.compute_angular_momentum_hessian(self.com)

    """
    Whole-body wrench
    =================
    """

    def compute_gravito_inertial_wrench(self, qdd, p):
        """
        Compute the gravito-inertial wrench at point `P`:

        .. math::

            w_P = \\left[\\begin{array}{c}
                f^{gi} \\\\
                \\tau^{gi}_P
            \\end{array}\\right]
            = \\left[\\begin{array}{c}
                m (g - \\ddot{p}_G) \\\\
                (p_G - p_P) \\times m (g - \\ddot{p}_G) - \\dot{L}_G
                \\end{array}\\right]

        with `m` the robot mass, `g` the gravity vector, `G` the center of
        mass, :math:`\\ddot{p}_G` the COM acceleration, and :math:`\\dot{L}_G`
        the rate of change of the centroidal angular momentum.

        Parameters
        ----------
        qdd : array
            Vector of DOF accelerations.
        p : array, shape=(3,)
            Application point of the gravito-inertial wrench.

        Returns
        -------
        w_P : array, shape=(6,)
            Coordinates of the gravito-inertial wrench expressed at point P in
            the world frame.
        """
        f, tau_P = zeros(3), zeros(3)
        link_velocities = self.rave.GetLinkVelocities()
        link_accelerations = self.rave.GetLinkAccelerations(qdd)
        for link in self.rave.GetLinks():
            I_local = link.GetLocalInertia()
            R = link.GetTransform()[0:3, 0:3]
            c = link.GetGlobalCOM()
            i = link.GetIndex()
            m = link.GetMass()
            omega = link_velocities[i][3:]
            omegad = link_accelerations[i][3:]
            pdd = link_accelerations[i][:3]
            r = dot(R, link.GetLocalCOM())
            cdd = pdd + cross(omega, cross(omega, r)) + cross(omegad, r)
            f_link = m * (gravity - cdd)
            L_local = dot(I_local, omegad) - cross(dot(I_local, omega), omega)
            f += f_link
            tau_P += cross(c - p, f_link) - dot(R, L_local)
        return hstack([f, tau_P])

    def compute_net_contact_wrench(self, qdd, p):
        """
        Compute the gravito-inertial wrench at point `P`:

        .. math::

            w_P = \\left[\\begin{array}{c}
                f^{c} \\\\
                \\tau^{c}_P
            \\end{array}\\right]
            = \\sum_i \\left[\\begin{array}{c}
                f_i \\\\
                (p_i - p_P) \\times f_i + \\tau_i
                \\end{array}\\right]

        with the contact wrench :math:`(f_i, \\tau_i)` is applied at the
        :math:`i^\\mathrm{th}` contact point located at :math:`p_i`.

        Parameters
        ----------
        qdd : array
            Vector of DOF accelerations.
        p : array, shape=(3,)
            Application point of the gravito-inertial wrench.

        Returns
        -------
        w_P : array, shape=(6,)
            Coordinates of the gravito-inertial wrench expressed at point P in
            the world frame.

        Notes
        -----
        From the `Newton-Euler equations
        <https://scaron.info/teaching/newton-euler-equations.html>`_ of the
        system, the net contact wrench is opposite to the gravito-inertial
        wrench computed by
        :func:`pymanoid.robot.Humanoid.compute_gravito_inertial_wrench`.
        """
        return -self.compute_gravito_inertial_wrench(qdd, p)

    def compute_zmp(self, qdd, origin=None, normal=None):
        """
        Compute the Zero-tilting Moment Point (ZMP).

        Parameters
        ----------
        qdd : array
            Vector of joint accelerations.
        origin : array, shape=(3,), optional
            Origin `O` of the ZMP plane. Defaults to the origin of the world
            frame.
        normal : array, shape=(3,), optional
            Normal `n` of the ZMP plane. Defaults to the vertical.

        Returns
        -------
        zmp : array, shape=(3,)
            ZMP of the gravito-inertial wrench in the plane (`O`, `n`).

        Notes
        -----
        See [Sardain04]_ for an excellent introduction to the concepts of ZMP
        and center of pressure. See [Caron17z]_ for the more general definition
        of ZMP support areas in arbitrary planes.
        """
        p_O = origin if origin is not None else zeros(3)
        n = normal if normal is not None else array([0, 0, 1])
        f, tau_O = self.compute_gravito_inertial_wrench(qdd, p_O)
        return cross(n, tau_O) / dot(n, f) + p_O
