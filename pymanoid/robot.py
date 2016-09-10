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
from ik import VelocitySolver
from numpy import concatenate, eye, maximum, minimum, ones, vstack, zeros
from os.path import basename, splitext
from threading import Lock, Thread
from time import sleep as rt_sleep
from warnings import warn


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
        env = get_env()
        env.LoadData(xml)
        rave = env.GetRobot(name)
        nb_dofs = rave.GetDOF()
        set_default_background_color()  # [dirty] reset by LoadData
        q_min, q_max = rave.GetDOFLimits()
        rave.SetDOFVelocities([0] * nb_dofs)
        rave.SetDOFVelocityLimits([1000.] * nb_dofs)
        if qd_lim is None:
            qd_lim = 10.  # [rad] / [s]; this is already quite fast
        qd_max = +qd_lim * ones(nb_dofs)
        qd_min = -qd_lim * ones(nb_dofs)

        self.active_dofs = None
        self.has_free_flyer = False
        self.ik = None  # created by self.init_ik()
        self.ik_lock = None
        self.ik_thread = None
        self.is_visible = True
        self.mass = sum([link.GetMass() for link in rave.GetLinks()])
        self.nb_dofs = nb_dofs
        self.q_max = q_max
        self.q_max.flags.writeable = False
        self.q_max_active = None
        self.q_min = q_min
        self.q_min.flags.writeable = False
        self.q_min_active = None
        self.qd_max = qd_max
        self.qd_max_active = None
        self.qd_min = qd_min
        self.qd_min_active = None
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
        self.rave.SetDOFLimits(q_min, q_max, dof_indices)
        self.q_min.flags.writeable = True
        self.q_max.flags.writeable = True
        if dof_indices is not None:
            assert len(q_min) == len(q_max) == len(dof_indices)
            self.q_min[dof_indices] = q_min
            self.q_max[dof_indices] = q_max
        else:
            assert len(q_min) == len(q_max) == self.nb_dofs
            self.q_min = q_min
            self.q_max = q_max
        if self.active_dofs:
            self.q_min_active = self.q_min[self.active_dofs]
            self.q_max_active = self.q_max[self.active_dofs]
        self.q_min.flags.writeable = False
        self.q_max.flags.writeable = False

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

        - ``link`` -- link index or pymanoid.Link object
        - ``p`` -- link frame origin (optional: if None, link.p is used)
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        H_lin = self.rave.ComputeHessianTranslation(link_index, p)
        H_ang = self.rave.ComputeHessianAxisAngle(link_index)
        H = concatenate([H_lin, H_ang], axis=1)
        return H

    """
    Inverse Kinematics
    ==================
    """

    def init_ik(self, gains=None, weights=None):
        """
        Initialize the IK solver.

        INPUT:

        - ``gains`` -- dictionary of default task gains
        - ``weights`` -- dictionary of default task weights
        """
        self.ik = VelocitySolver(
            self, default_gains=gains, default_weights=weights)

    def step_ik(self, dt):
        qd_active = self.ik.compute_velocity(dt)
        q_active = minimum(
            maximum(
                self.q_min_active,
                self.q_active + qd_active * dt),
            self.q_max_active)
        self.set_active_dof_values(q_active)
        self.set_active_dof_velocities(qd_active)

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

    def start_ik_thread(self, dt, sleep_fun=None):
        """
        Start a new thread stepping the IK every dt, then calling sleep_fun(dt).

        INPUT:

        - ``dt`` -- stepping period in seconds
        - ``sleep_fun`` -- sleeping function (default: time.sleep)
        """
        if sleep_fun is None:
            sleep_fun = rt_sleep
        self.ik_lock = Lock()
        self.ik_thread = Thread(target=self.run_ik_thread, args=(dt, sleep_fun))
        self.ik_thread.daemon = True
        self.ik_thread.start()

    def run_ik_thread(self, dt, sleep_fun):
        while self.ik_lock:
            with self.ik_lock:
                self.step_ik(dt)
                sleep_fun(dt)

    def pause_ik_thread(self):
        self.ik_lock.acquire()

    def resume_ik_thread(self):
        self.ik_lock.release()

    def stop_ik_thread(self):
        self.ik_lock = None

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
