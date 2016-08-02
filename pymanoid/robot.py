#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
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


from contact import Contact, ContactSet
from env import get_env, get_viewer
from env import set_default_background_color
from numpy import arange, array, concatenate, cross, dot, eye, maximum, minimum
from numpy import zeros, hstack, vstack, tensordot, ndarray
from openravepy import RaveCreateModule
from os.path import basename, splitext
from rotations import crossmat
from time import sleep as rt_sleep
from threading import Lock, Thread
from warnings import warn


_oppose_quat = array([-1., -1., -1., -1., +1., +1., +1.])


# Notations and names
# ===================
#
# am: Angular Momentum
# am_rate: Rate (time-derivative) of Angular Momentum
# c: link COM
# m: link mass
# omega: link angular velocity
# r: origin of link frame
# R: link rotation
# T: link transform
# v: link velocity (v = [rd, omega])
#
# Unless otherwise mentioned, coordinates are in the absolute reference frame.

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

    def __init__(self, path, root_body, free_flyer=True):
        """
        Create a new robot object.

        INPUT:

        - ``path`` -- path to the COLLADA model of the robot

        """
        name = basename(splitext(path)[0])
        if free_flyer:
            xml = Robot.__free_flyer_xml % (path, name, root_body)
        else:
            xml = Robot.__default_xml % (path, name)
        env = get_env()
        env.LoadData(xml)
        set_default_background_color()  # reset by LoadData
        robot = env.GetRobot(name)
        q_min, q_max = robot.GetDOFLimits()
        robot.SetDOFVelocityLimits(1000 * robot.GetDOFVelocityLimits())
        robot.SetDOFVelocities([0] * robot.GetDOF())

        self.active_dofs = None
        self.has_free_flyer = free_flyer
        self.ik = None  # created by self.init_ik()
        self.ik_lock = None
        self.ik_thread = None
        self.is_visible = True
        self.mass = sum([link.GetMass() for link in robot.GetLinks()])
        self.q_max_full = q_max
        self.q_max_full.flags.writeable = False
        self.q_min_full = q_min
        self.q_min_full.flags.writeable = False
        self.qdd_max_full = None  # set in child class
        self.rave = robot
        self.tau_max_full = None  # set by hand in child robot class
        self.transparency = 0.  # initially opaque

    #
    # Properties
    #

    @property
    def cam(self):
        return self.compute_cam(self.q, self.qd)

    @property
    def com(self):
        return self.compute_com(self.q)

    @property
    def comd(self):
        return self.compute_com_velocity(self.q, self.qd)

    @property
    def nb_active_dofs(self):
        return self.rave.GetActiveDOF()

    @property
    def nb_dofs(self):
        return self.rave.GetDOF()

    @property
    def q(self):
        return self.get_dof_values()

    @property
    def q_full(self):
        return self.rave.GetDOFValues()

    @property
    def q_max(self):
        if not self.active_dofs:
            return self.q_max_full
        return self.q_max_full[self.active_dofs]

    @property
    def q_min(self):
        if not self.active_dofs:
            return self.q_min_full
        return self.q_min_full[self.active_dofs]

    @property
    def qd(self):
        return self.get_dof_velocities()

    @property
    def qd_full(self):
        return self.rave.GetDOFVelocities()

    @property
    def qdd_max(self):
        assert self.qdd_max_full is not None, "Acceleration limits unset"
        if not self.active_dofs:
            return self.qdd_max_full
        return self.qdd_max_full[self.active_dofs]

    @property
    def tau_max(self):
        assert self.tau_max_full is not None, "Torque limits unset"
        if not self.active_dofs:
            return self.tau_max_full
        return self.tau_max_full[self.active_dofs]

    #
    # Kinematics
    #

    def set_active_dofs(self, active_dofs):
        """
        Set active DOFs.

        active_dofs -- list of DOF indices
        """
        self.active_dofs = active_dofs
        self.rave.SetActiveDOFs(active_dofs)

    def init_ik(self, gains=None, weights=None, qd_lim=None, K_doflim=None):
        """
        Initialize the IK solver. Needs to be defined by child classes.

        INPUT:

        ``gain`` -- dictionary of default task gains
        ``weights`` -- dictionary of default task weights
        ``qd_lim`` -- special gain to enforce velocity limits
        ``K_doflim`` -- special gain to enforce DOF limits

        EXAMPLE:

            self.ik = DiffIKSolver(
                dt=dt,
                q_max=self.q_max,
                q_min=self.q_min,
                qd_lim=10.,
                K_doflim=5.,
                gains={
                    'com': 1.,
                    'contact': 0.9,
                    'link': 0.2,
                    'posture': 0.005,
                },
                weights={
                    'contact': 100.,
                    'com': 5.,
                    'link': 5.,
                    'posture': 0.1,
                })

        """
        warn("no IK defined for robot of type %s" % (str(type(self))))

    def get_dof_values(self, dof_indices=None):
        if dof_indices is not None:
            return self.rave.GetDOFValues(dof_indices)
        elif self.active_dofs:
            return self.rave.GetActiveDOFValues()
        return self.rave.GetDOFValues()

    def set_dof_values(self, q, dof_indices=None):
        if dof_indices is not None:
            return self.rave.SetDOFValues(q, dof_indices)
        elif self.active_dofs and len(q) == self.nb_active_dofs:
            return self.rave.SetActiveDOFValues(q)
        assert len(q) == self.nb_dofs, "Invalid DOF vector"
        return self.rave.SetDOFValues(q)

    def update_dof_limits(self, dof_index, q_min=None, q_max=None):
        if q_min is not None:
            self.q_min_full.flags.writeable = True
            self.q_min_full[dof_index] = q_min
            self.q_min_full.flags.writeable = False
        if q_max is not None:
            self.q_max_full.flags.writeable = True
            self.q_max_full[dof_index] = q_max
            self.q_max_full.flags.writeable = False

    def scale_dof_limits(self, scale=1.):
        q_avg = .5 * (self.q_max_full + self.q_min_full)
        q_dev = .5 * (self.q_max_full - self.q_min_full)
        self.q_max_full.flags.writeable = True
        self.q_min_full.flags.writeable = True
        self.q_max_full = (q_avg + scale * q_dev)
        self.q_min_full = (q_avg - scale * q_dev)
        self.q_max_full.flags.writeable = False
        self.q_min_full.flags.writeable = False

    def get_dof_velocities(self, dof_indices=None):
        if dof_indices is not None:
            return self.rave.GetDOFVelocities(dof_indices)
        elif self.active_dofs:
            return self.rave.GetActiveDOFVelocities()
        return self.rave.GetDOFVelocities()

    def set_dof_velocities(self, qd, dof_indices=None):
        check_dof_limits = 0  # CLA_Nothing
        if dof_indices is not None:
            return self.rave.SetDOFVelocities(qd, check_dof_limits, dof_indices)
        elif self.active_dofs and len(qd) == self.nb_active_dofs:
            return self.rave.SetActiveDOFVelocities(qd, check_dof_limits)
        assert len(qd) == self.nb_dofs, "Invalid DOF velocity vector"
        return self.rave.SetDOFVelocities(qd)

    #
    # Inverse Kinematics
    #

    def add_com_task(self, target, gain=None, weight=None):
        if type(target) is list:
            target = array(target)
        if type(target) is ndarray:
            def error(q, qd, dt):
                return target - self.compute_com(q)
        elif hasattr(target, 'pos'):
            def error(q, qd, dt):
                return target.pos - self.compute_com(q)
        elif hasattr(target, 'p'):
            def error(q, qd, dt):
                return target.p - self.compute_com(q)
        else:  # COM target should be a position
            msg = "Target of type %s has no 'pos' attribute" % type(target)
            raise Exception(msg)

        jacobian = self.compute_com_jacobian
        self.ik.add_task('com', error, jacobian, gain, weight)

    def add_constant_cam_task(self, weight=None):
        """
        Try to keep the centroidal angular momentum constant.

        INPUT:

        ``weight`` -- task weight (optional)

        .. NOTE::

            The way this task is implemented may be surprising. Basically,
            keeping a constant CAM means d/dt(CAM) == 0, i.e.,

                d/dt (J_cam * qd) == 0
                J_cam * qdd + qd * H_cam * qd == 0

            Because the IK works at the velocity level, we approximate qdd by
            finite difference from the previous velocity (``qd`` argument to the
            error function):

                J_cam * (qd_next - qd) / dt + qd * H_cam * qd == 0

            Finally, the task in qd_next (output velocity) is:

                J_cam * qd_next == J_cam * qd - dt * qd * H_cam * qd

            Hence, there are two occurrences of J_cam: one in the task error,
            and the second in the task jacobian.

        """
        def error(q, qd, dt):
            J_cam = self.compute_cam_jacobian(q)
            H_cam = self.compute_cam_hessian(q)  # computation intensive :(
            return dot(J_cam, qd) - dt * dot(qd, dot(H_cam, qd))

        jacobian = self.compute_cam_jacobian
        self.ik.add_task('cam', error, jacobian, weight=weight, unit_gain=True)

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
        ``weight`` -- multiplier of the task squared error in IK cost function
        ``alpha`` -- callable function returning a gain multiplier (float)
        """
        def error(q, qd, dt):
            cur_pose = self.compute_link_pose(link, q)
            return alpha() * (contact.effector_pose - cur_pose)

        def jacobian(q):
            return self.compute_link_active_pose_jacobian(link, q)

        task_name = 'contact-%s' % link.name
        self.ik.add_task(task_name, error, jacobian, gain, weight)

    def add_dof_task(self, dof_id, dof_ref, gain=None, weight=None):
        active_dof_id = self.active_dofs.index(dof_id)
        J = zeros(self.nb_active_dofs)
        J[active_dof_id] = 1.

        def error(q, qd, dt):
            return (dof_ref - q[active_dof_id])

        def jacobian(q):
            return J

        task_name = 'dof-%d' % dof_id
        self.ik.add_task(task_name, error, jacobian, gain, weight)

    def compute_link_pose_error(self, link, q, target_pose):
        link_pose = self.compute_link_pose(link, q)
        diff = target_pose - link_pose
        if dot(diff[0:4], diff[0:4]) > 1.:
            return _oppose_quat * target_pose - link_pose
        return diff

    def add_link_pose_task(self, link, target, gain=None, weight=None,
                           task_type='link'):
        if type(target) is Contact:  # used for ROS communications
            target.robot_link = link.index  # dirty
        if type(target) is list:
            target = array(target)
        if hasattr(target, 'effector_pose'):  # needs to come before 'pose'
            def error(q, qd, dt):
                return self.compute_link_pose_error(
                    link, q, target.effector_pose)
        elif hasattr(target, 'pose'):
            def error(q, qd, dt):
                return self.compute_link_pose_error(link, q, target.pose)
        elif type(target) is ndarray:
            def error(q, qd, dt):
                return self.compute_link_pose_error(link, q, target)
        else:  # link frame target should be a pose
            msg = "Target of type %s has no 'pose' attribute" % type(target)
            raise Exception(msg)

        def jacobian(q):
            return self.compute_link_active_pose_jacobian(link, q)

        self.ik.add_task(link.name, error, jacobian, gain, weight, task_type)

    def add_link_position_task(self, link, target, gain=None, weight=None):
        if type(target) is list:
            target = array(target)
        if hasattr(target, 'pos'):
            def error(q, qd, dt):
                return target.pos - link.p
        elif hasattr(target, 'p'):
            def error(q, qd, dt):
                return target.p - link.p
        elif type(target) is ndarray:
            def error(q, qd, dt):
                return target - self.compute_link_pose(link, q)
        else:  # this is an aesthetic comment
            msg = "Target of type %s has no 'pos' attribute" % type(target)
            raise Exception(msg)

        def jacobian(q):
            return self.compute_link_active_position_jacobian(link, q)

        task_name = 'link-%s' % link.name
        self.ik.add_task(task_name, error, jacobian, gain, weight)

    def add_min_acceleration_task(self, weight):
        """
        Minimize accelerations.

        INPUT:

        ``weight`` -- task weight

        .. NOTE::

            As the differential IK returns velocities, we approximate the task
            "minimize qdd" by "minimize (qd_next - qd)".
        """
        identity = eye(self.nb_active_dofs)

        def error(q, qd, dt):
            return qd

        def jacobian(q):
            return identity

        self.ik.add_task('qdd-min', error, jacobian, weight=weight,
                         unit_gain=True)

    def add_min_velocity_task(self, gain=None, weight=None):
        """
        Minimize the instantaneous velocity.

        INPUT:

        ``gain`` -- task gain
        ``weight`` -- task weight

        .. NOTE::

            This is a regularization task, therefore ``weight`` should be low
            compared to the other task weights.
        """
        identity = eye(self.nb_active_dofs)

        def error(q, qd, dt):
            return -qd

        def jacobian(q):
            return identity

        self.ik.add_task('qdmin', error, jacobian, gain, weight)

    def add_posture_task(self, q_ref, gain=None, weight=None):
        """
        Add a postural task, a common choice to regularize the weighted IK
        problem.

        INPUT:

        ``gain`` -- task gain
        ``weight`` -- task weight

        .. NOTE::

            This is a regularization task, therefore ``weight`` should be low
            compared to the other task weights.
        """
        if len(q_ref) == self.nb_dofs:
            q_ref = q_ref[self.active_dofs]

        J_posture = eye(self.nb_active_dofs)
        trans = []

        if self.has_free_flyer:  # don't include translation coordinates
            for i in [self.TRANS_X, self.TRANS_Y, self.TRANS_Z]:
                if i in self.active_dofs:
                    trans.append(self.active_dofs.index(i))
            for j in trans:
                J_posture[j, j] = 0.

        def error(q, qd, dt):
            e = (q_ref - q)
            for j in trans:
                e[j] = 0.
            return e

        def jacobian(q):
            return J_posture

        self.ik.add_task('posture', error, jacobian, gain, weight)

    def add_min_cam_task(self, weight=None):
        """
        Minimize the centroidal angular momentum.

        INPUT:

        ``weight`` -- task weight (optional)
        """
        def error(q, qd, dt):
            return zeros((3,))

        jacobian = self.compute_cam_jacobian
        self.ik.add_task('cam', error, jacobian, weight=weight, unit_gain=True)

    def remove_contact_task(self, link):
        self.ik.remove_task(link.name)

    def remove_link_task(self, link):
        self.ik.remove_task(link.name)

    def remove_dof_task(self, dof_id):
        self.ik.remove_task('dof-%d' % dof_id)

    def update_com_task(self, target, gain=None, weight=None):
        if 'com' not in self.ik.gains or 'com' not in self.ik.weights:
            raise Exception("No COM task to update in robot IK")
        gain = self.ik.gains['com']
        weights = self.ik.weights['com']
        self.ik.remove_task('com')
        self.add_com_task(target, gain, weight)

    def step_ik(self, dt):
        qd = self.ik.compute_velocity(self.q, self.qd, dt)
        q = minimum(maximum(self.q_min, self.q + qd * dt), self.q_max)
        self.set_dof_values(q)
        self.set_dof_velocities(qd)

    def solve_ik(self, max_it=100, conv_tol=1e-5, dt=1e-2, debug=False):
        """
        Compute joint-angles q satisfying all constraints at best.

        INPUT:

        - ``max_it`` -- maximum number of differential IK iterations
        - ``conv_tol`` -- stop when the cost improvement ratio is less than this
            threshold
        - ``dt`` -- time step for the differential IK
        - ``debug`` -- print extra debug info

        .. NOTE::

            Good values of dt depend on the gains and weights of the IK tasks.
            Small values make convergence slower, while big values will render
            them unstable.
        """
        cur_cost = 1000.
        q, qd = self.q, zeros(self.nb_active_dofs)
        # qd = zeros(len(self.q))
        if debug:
            print "solve_ik(max_it=%d, conv_tol=%e)" % (max_it, conv_tol)
        for itnum in xrange(max_it):
            prev_cost = cur_cost
            cur_cost = self.ik.compute_cost(q, qd, dt)
            cost_var = cur_cost - prev_cost
            if debug:
                print "%2d: %.3f (%+.2e)" % (itnum, cur_cost, cost_var)
            if abs(cost_var) < conv_tol:
                if abs(cur_cost) > 0.1:
                    warn("IK did not converge to solution. Is it feasible?"
                         "If so, try restarting from a random guess.")
                break
            qd = self.ik.compute_velocity(q, qd, dt)
            q = minimum(maximum(self.q_min, q + qd * dt), self.q_max)
            if debug:
                self.set_dof_values(q)
        self.set_dof_values(q)
        return itnum, cur_cost

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

    #
    # IK Threading
    #

    def start_ik_thread(self, dt, sleep_fun=None):
        """
        Start a new thread stepping the IK every dt, then calling sleep_fun(dt).

        dt -- stepping period in seconds
        sleep_fun -- sleeping function (default: time.sleep)
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

    #
    # Visualization
    #

    def play_trajectory(self, traj, callback=None, dt=3e-2):
        trange = list(arange(0, traj.T, dt))
        for t in trange:
            q = traj.q(t)
            qd = traj.qd(t)
            qdd = traj.qdd(t)
            self.set_dof_values(q)
            if callback:
                callback(t, q, qd, qdd)
            rt_sleep(dt)

    def record_trajectory(self, traj, fname='output.mpg', codec=13,
                          framerate=24, width=800, height=600, dt=3e-2):
        env = get_env()
        viewer = get_viewer()
        recorder = RaveCreateModule(env, 'viewerrecorder')
        env.AddModule(recorder, '')
        self.set_dof_values(traj.q(0))
        recorder.SendCommand('Start %d %d %d codec %d timing '
                             'simtime filename %s\n'
                             'viewer %s' % (width, height, framerate, codec,
                                            fname, viewer.GetName()))
        rt_sleep(1.)
        self.play_trajectory(traj, dt=dt)
        rt_sleep(1.)
        recorder.SendCommand('Stop')
        env.Remove(recorder)

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

    def show(self):
        self.rave.SetVisible(True)

    def hide(self):
        self.rave.SetVisible(False)

    def set_visible(self, visible):
        self.is_visible = visible
        self.rave.SetVisible(visible)

    #
    # Positions
    #

    def compute_com(self, q):
        total = zeros(3)
        with self.rave:
            self.set_dof_values(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                c = link.GetGlobalCOM()
                total += m * c
        return total / self.mass

    def compute_link_pos(self, link, q, link_coord=None):
        with self.rave:
            self.set_dof_values(q)
            T = link.T
            if link_coord is None:
                return T[:3, 3]
            return dot(T, hstack([link_coord, 1]))[:3]

    def compute_link_pose(self, link, q=None):
        if q is None:
            return link.pose
        with self.rave:
            self.set_dof_values(q)
            return link.pose  # first coefficient will be positive

    #
    # Velocities
    #

    def compute_angular_momentum(self, q, qd, p):
        """
        Compute the angular momentum with respect to point p.

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point in world coordinates
        """
        am = zeros(3)
        with self.rave:
            self.set_dof_values(q)
            self.set_dof_velocities(qd)
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

    def compute_cam(self, q, qd):
        """
        Compute the centroidal angular momentum (CAM), that is to say, the
        angular momentum taken at the COM in the world frame.
        """
        return self.compute_angular_momentum(q, qd, self.compute_com(q))

    def compute_com_velocity(self, q, qd):
        total = zeros(3)
        with self.rave:
            self.set_dof_values(q)
            self.set_dof_velocities(qd)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                R = link.GetTransform()[0:3, 0:3]
                c_local = link.GetLocalCOM()
                v = link.GetVelocity()
                rd, omega = v[:3], v[3:]
                cd = rd + cross(omega, dot(R, c_local))
                total += m * cd
        return total / self.mass

    #
    # Accelerations
    #

    def compute_cam_rate(self, q, qd, qdd):
        J = self.compute_cam_jacobian(q)
        H = self.compute_cam_hessian(q)
        return dot(J, qdd) + dot(qd, dot(H, qd))

    def compute_com_acceleration(self, q, qd, qdd):
        J = self.compute_com_jacobian(q)
        H = self.compute_com_hessian(q)
        return dot(J, qdd) + dot(qd, dot(H, qdd))

    def compute_gravito_inertial_wrench(self, q, qd, qdd, p):
        """
        Compute the gravito-inertial wrench

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
        with self.rave:
            self.set_dof_values(q)
            self.set_dof_velocities(qd)
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

    def compute_zmp(self, q, qd, qdd):
        """
        Compute the Zero-tilting Moment Point (ZMP).

        The best introduction I know to this concept is:

            P. Sardain and G. Bessonnet, “Forces acting on a biped robot. center
            of pressure-zero moment point,” Systems, Man and Cybernetics, Part
            A: Systems and Humans, IEEE Transactions on, vol. 34, no. 5, pp.
            630– 637, 2004.

        q -- array of DOF values
        qd -- array of DOF velocities
        qdd -- array of DOF accelerations
        """
        O, n = zeros(3), array([0, 0, 1])
        f_gi, tau_gi = self.compute_gravito_inertial_wrench(q, qd, qdd, O)
        return cross(n, tau_gi) * 1. / dot(n, f_gi)

    #
    # Jacobians
    #

    def compute_angular_momentum_jacobian(self, q, p):
        """
        Compute the jacobian matrix J(q) such that the angular momentum of the
        robot at p is given by:

            L_p(q, qd) = J(q) * qd

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point in world coordinates
        """
        J_am = zeros((3, self.nb_dofs))
        with self.rave:
            self.set_dof_values(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                i = link.GetIndex()
                c = link.GetGlobalCOM()
                R = link.GetTransform()[0:3, 0:3]
                I = dot(R, dot(link.GetLocalInertia(), R.T))
                J_trans = self.rave.ComputeJacobianTranslation(i, c)
                J_rot = self.rave.ComputeJacobianAxisAngle(i)
                J_am += dot(crossmat(c - p), m * J_trans) + dot(I, J_rot)
        if self.active_dofs and len(q) == self.nb_active_dofs:
            return J_am[:, self.active_dofs]
        return J_am

    def compute_cam_jacobian(self, q):
        """
        Compute the jacobian matrix J(q) such that the centroidal angular
        momentum (i.e. the angular momentum of the robot at its center of mass
        G) is given by:

            L_G(q, qd) = J(q) * qd

        q -- vector of joint angles
        qd -- vector of joint velocities
        L_G -- angular momentum at the center of mass G
        """
        p_G = self.compute_com(q)
        return self.compute_angular_momentum_jacobian(q, p_G)

    def compute_com_jacobian(self, q):
        """
        Compute the jacobian matrix J(q) of the position of the center of mass G
        of the robot, i.e. the velocity of the G is given by:

                pd_G(q, qd) = J(q) * qd

        q -- vector of joint angles
        qd -- vector of joint velocities
        pd_G -- velocity of the center of mass G
        """
        J_com = zeros((3, self.nb_dofs))
        with self.rave:
            self.set_dof_values(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                if m < 1e-4:
                    continue
                index = link.GetIndex()
                c = link.GetGlobalCOM()
                J_com += m * self.rave.ComputeJacobianTranslation(index, c)
            J_com /= self.mass
        if self.active_dofs and len(q) == self.nb_active_dofs:
            return J_com[:, self.active_dofs]
        return J_com

    def compute_link_jacobian(self, link, p=None, q=None):
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
        with self.rave:
            if q is not None:
                self.set_dof_values(q)
            J_lin = self.rave.ComputeJacobianTranslation(link_index, p)
            J_ang = self.rave.ComputeJacobianAxisAngle(link_index)
            J = vstack([J_lin, J_ang])
        return J

    def compute_link_active_jacobian(self, link, q):
        J = self.compute_link_jacobian(link, q)
        return J[:, self.active_dofs]

    def compute_link_pose_jacobian(self, link, q=None):
        with self.rave:
            self.set_dof_values(q)
            J_trans = self.rave.CalculateJacobian(link.index, link.p)
            or_quat = link.rave.GetTransformPose()[:4]  # don't use link.pose
            J_quat = self.rave.CalculateRotationJacobian(link.index, or_quat)
            if or_quat[0] < 0:  # we enforce positive first coefficients
                J_quat *= -1.
            J = vstack([J_quat, J_trans])
        return J

    def compute_link_active_pose_jacobian(self, link, q=None):
        J = self.compute_link_pose_jacobian(link, q)
        return J[:, self.active_dofs]

    def compute_link_position_jacobian(self, link, p=None, q=None):
        """
        Compute the position Jacobian of a point p on a given robot link.

        link -- link index or pymanoid.Link object
        p -- point coordinates in world frame (optional)
        q -- DOF values to compute the jacobian at (optional)

        When p is None, the origin of the link reference frame is taken.
        When q is None, the current robot's DOF values are used.
        """
        link_index = link if type(link) is int else link.index
        with self.rave:
            if q is not None:
                self.set_dof_values(q)
            p = link.p if p is None else p
            J = self.rave.ComputeJacobianTranslation(link_index, p)
        return J

    def compute_link_active_position_jacobian(self, link, p=None, q=None):
        J = self.compute_link_position_jacobian(link, p, q)
        return J[:, self.active_dofs]

    #
    # Hessians
    #

    def compute_link_hessian(self, link, p=None, q=None):
        """
        Compute the hessian H(q) of the reference frame of the link, i.e. the
        acceleration of the link frame is given by:

            [a omegad] = J(q) * qdd + qd.T * H(q) * qd

        where a and omegad are the linear and angular accelerations of the
        frame, and J(q) is the frame jacobian.

        link -- link index or pymanoid.Link object
        p -- link frame origin (optional: if None, link.p is used)
        q -- vector of joint angles (optional: if None, robot.q is used)
        """
        link_index = link if type(link) is int else link.index
        p = p if type(link) is int else link.p
        with self.rave:
            if q is not None:
                self.set_dof_values(q)
            H_lin = self.rave.ComputeHessianTranslation(link_index, p)
            H_ang = self.rave.ComputeHessianAxisAngle(link_index)
            H = concatenate([H_lin, H_ang], axis=1)
        return H

    def compute_link_active_frame_hessian(self, link, q):
        H = self.compute_link_hessian(link, q)
        return H[:, self.active_dofs]

    def compute_angular_momentum_hessian(self, q, p):
        """
        Returns a matrix H(q) such that the rate of change of the angular
        momentum with respect to point p is

            Ld_p(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd)),

        where J(q) is the angular-momentum jacobian.

        q -- joint angle values
        qd -- joint-angle velocities
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
        with self.rave:
            self.set_dof_values(q)
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
        if self.active_dofs and len(q) == self.nb_active_dofs:
            return ((H[self.active_dofs, :, :])[:, :, self.active_dofs])
        return H

    def compute_cam_hessian(self, q):
        """
        Returns a matrix H(q) such that the rate of change of the angular
        momentum with respect to the center of mass G is

            Ld_G(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd)),

        q -- joint angle values
        """
        p_G = self.compute_com(q)
        return self.compute_angular_momentum_hessian(q, p_G)

    def compute_com_hessian(self, q):
        H_com = zeros((self.nb_dofs, 3, self.nb_dofs))
        with self.rave:
            self.set_dof_values(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                if m < 1e-4:
                    continue
                index = link.GetIndex()
                c = link.GetGlobalCOM()
                H_com += m * self.rave.ComputeHessianTranslation(index, c)
            H_com /= self.mass
        if self.active_dofs and len(q) == self.nb_active_dofs:
            return ((H_com[self.active_dofs, :, :])[:, :, self.active_dofs])
        return H_com

    #
    # Dynamics
    #

    def compute_inertia_matrix(self, q=None, external_torque=None):
        """
        Compute the inertia matrix of the robot, that is, the matrix M(q) such
        that the equations of motion are:

            M(q) * qdd + qd.T * C(q) * qd + g(q) = F + external_torque

        with:

        q -- vector of joint angles (DOF values)
        qd -- vector of joint velocities
        qdd -- vector of joint accelerations
        C(q) -- Coriolis tensor (derivative of M(q) w.r.t. q)
        g(q) -- gravity vector
        F -- vector of generalized forces (joint torques, contact wrenches, ...)
        external_torque -- additional torque vector (optional)

        This function applies the unit-vector method described by Walker & Orin
        <https://dx.doi.org/10.1115/1.3139699>.
        """
        M = zeros((self.nb_dofs, self.nb_dofs))
        with self.rave:
            if q is not None:
                self.set_dof_values(q)
            for (i, e_i) in enumerate(eye(self.nb_dofs)):
                tm, _, _ = self.rave.ComputeInverseDynamics(
                    e_i, external_torque, returncomponents=True)
                M[:, i] = tm
        return M

    def compute_inverse_dynamics(self, q=None, qd=None, qdd=None,
                                 external_torque=None):
        """
        Wrapper around OpenRAVE's ComputeInverseDynamics function, which
        implements the Recursive Newton-Euler algorithm by Walker & Orin
        <https://dx.doi.org/10.1115/1.3139699>.

        The function returns three terms tm, tc and tg such that

            tm = M(q) * qdd
            tc = qd.T * C(q) * qd
            tg = g(q)

        and the equations of motion are written:

            tm + tc + tg = F + external_torque

        with:

        q -- vector of joint angles (DOF values)
        qd -- vector of joint velocities
        qdd -- vector of joint accelerations
        M(q) -- inertia matrix
        C(q) -- Coriolis tensor (derivative of M(q) w.r.t. q)
        g(q) -- gravity vector
        F -- vector of generalized forces (joint torques, contact wrenches, ...)
        external_torque -- additional torque vector (optional)

        If q or qd are not given, the robot's current DOF values/velocities are
        used. If qdd is not given, the return value for tm will be None.
        """
        with self.rave:
            if q is not None:
                self.set_dof_values(q)
            if qd is not None:
                self.set_dof_velocities(qd)
            if qdd is None:
                _, tc, tg = self.rave.ComputeInverseDynamics(
                    zeros(self.nb_dofs), external_torque, returncomponents=True)
                return None, tc, tg
            tm, tc, tg = self.rave.ComputeInverseDynamics(
                qdd, external_torque, returncomponents=True)
            return tm, tc, tg
