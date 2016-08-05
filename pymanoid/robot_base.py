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

from env import get_env
from env import set_default_background_color
from numpy import eye, ones, zeros
from os.path import basename, splitext


"""
Notations and names
===================

am: Angular Momentum
am_rate: Rate (time-derivative) of Angular Momentum
c: link COM
m: link mass
omega: link angular velocity
r: origin of link frame
R: link rotation
T: link transform
v: link velocity (v = [rd, omega])

Unless otherwise mentioned, coordinates are in the absolute reference frame.
"""


class Robot(object):

    __default_xml = """
    <environment>
        <robot file="%s" name="%s" />
    </environment>
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
                    <joint name="FLYER_TX" type="slider" circular="true">
                        <body>FLYER_TX_LINK</body>
                        <body>FLYER_TY_LINK</body>
                        <axis>1 0 0</axis>
                        <limits>-10 +10</limits>
                    </joint>
                    <joint name="FLYER_TY" type="slider" circular="true">
                        <body>FLYER_TY_LINK</body>
                        <body>FLYER_TZ_LINK</body>
                        <axis>0 1 0</axis>
                        <limits>-10 +10</limits>
                    </joint>
                    <joint name="FLYER_TZ" type="slider" circular="true">
                        <body>FLYER_TZ_LINK</body>
                        <body>FLYER_ROLL_LINK</body>
                        <axis>0 0 1</axis>
                        <limits>-10 +10</limits>
                    </joint>
                    <joint name="FLYER_ROLL" type="hinge" circular="true">
                        <body>FLYER_ROLL_LINK</body>
                        <body>FLYER_PITCH_LINK</body>
                        <axis>1 0 0</axis>
                    </joint>
                    <joint name="FLYER_PITCH" type="hinge" circular="true">
                        <body>FLYER_PITCH_LINK</body>
                        <body>FLYER_YAW_LINK</body>
                        <axis>0 1 0</axis>
                    </joint>
                    <joint name="FLYER_YAW" type="hinge" circular="true">
                        <body>FLYER_YAW_LINK</body>
                        <body>%s</body>
                        <axis>0 0 1</axis>
                    </joint>
                </kinbody>
            </robot>
        </robot>
    </environment>
    """

    def __init__(self, path, root_body, free_flyer=True, qd_lim=10.):
        """
        Create a new robot object.

        INPUT:

        - ``path`` -- path to the COLLADA model of the robot
        - ``root_body`` -- name of first body in COLLADA file
        - ``free_flyer`` -- add 6 unactuated DOF? (optional, default is True)
        - ``qd_lim`` -- maximum angular joint velocity (in [rad])
        """
        name = basename(splitext(path)[0])
        if free_flyer:
            xml = Robot.__free_flyer_xml % (path, name, root_body)
        else:
            xml = Robot.__default_xml % (path, name)
        env = get_env()
        env.LoadData(xml)
        set_default_background_color()  # reset by LoadData
        rave = env.GetRobot(name)
        q_min, q_max = rave.GetDOFLimits()
        rave.SetDOFVelocityLimits(1000 * rave.GetDOFVelocityLimits())
        rave.SetDOFVelocities([0] * rave.GetDOF())

        self.active_dofs = None
        self.has_free_flyer = free_flyer
        self.is_visible = True
        self.mass = sum([link.GetMass() for link in rave.GetLinks()])
        self.q_max = q_max
        self.q_max.flags.writeable = False
        self.q_max_active = None
        self.q_min = q_min
        self.q_min.flags.writeable = False
        self.q_min_active = None
        self.qd_max = +qd_lim * ones(len(q_max))
        self.qd_max_active = None
        self.qd_min = -qd_lim * ones(len(q_min))
        self.qd_min_active = None
        self.qdd_max = None  # set in child class
        self.rave = rave
        self.tau_max = None  # set by hand in child robot class
        self.transparency = 0.  # initially opaque

    """
    Degrees of freedom
    ==================

    OpenRAVE calls "DOF values" what we will also call "joint angles". Same for
    "DOF velocities" and "joint velocities".
    """

    @property
    def nb_dofs(self):
        return self.rave.GetDOF()

    @property
    def q(self):
        return self.rave.GetDOFValues()

    @property
    def qd(self):
        return self.rave.GetDOFVelocities()

    def get_dof_values(self, dof_indices=None):
        if dof_indices is not None:
            return self.rave.GetDOFValues(dof_indices)
        return self.rave.GetDOFValues()

    def get_dof_velocities(self, dof_indices=None):
        if dof_indices is not None:
            return self.rave.GetDOFVelocities(dof_indices)
        return self.rave.GetDOFVelocities()

    def set_dof_values(self, q, dof_indices=None):
        if dof_indices is not None:
            return self.rave.SetDOFValues(q, dof_indices)
        return self.rave.SetDOFValues(q)

    def set_dof_velocities(self, qd, dof_indices=None):
        check_dof_limits = 0  # CLA_Nothing
        if dof_indices is not None:
            return self.rave.SetDOFVelocities(qd, check_dof_limits, dof_indices)
        return self.rave.SetDOFVelocities(qd)

    """
    Active DOFs
    ===========

    We simply wrap around OpenRAVE here. Active DOFs are used with the IK.
    """

    @property
    def nb_active_dofs(self):
        return self.rave.GetActiveDOF()

    @property
    def q_active(self):
        return self.rave.GetActiveDOFValues()

    def get_active_dof_values(self):
        return self.rave.GetActiveDOFValues()

    def get_active_dof_velocities(self):
        return self.rave.GetActiveDOFVelocities()

    def set_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs
        self.rave.SetActiveDOFs(active_dofs)
        self.q_max_active = self.q_max[active_dofs]
        self.q_min_active = self.q_min[active_dofs]
        self.qd_max_active = self.qd_max[active_dofs]
        self.qd_min_active = self.qd_min[active_dofs]

    def set_active_dof_values(self, q_active):
        return self.rave.SetActiveDOFValues(q_active)

    def set_active_dof_velocities(self, qd_active):
        check_dof_limits = 0  # CLA_Nothing
        return self.rave.SetActiveDOFVelocities(qd_active, check_dof_limits)

    """
    DOF limits
    ==========
    """

    def scale_dof_limits(self, scale=1.):
        q_avg = .5 * (self.q_max + self.q_min)
        q_dev = .5 * (self.q_max - self.q_min)
        self.q_max.flags.writeable = True
        self.q_min.flags.writeable = True
        self.q_max = (q_avg + scale * q_dev)
        self.q_min = (q_avg - scale * q_dev)
        self.q_max.flags.writeable = False
        self.q_min.flags.writeable = False

    """
    Dynamics
    ========
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
