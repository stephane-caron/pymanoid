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

from numpy import eye, maximum, minimum, zeros
from robot_base import BaseRobot
from threading import Lock, Thread
from time import sleep as rt_sleep
from warnings import warn


class IKRobot(BaseRobot):

    """
    Robot with an Inverse Kinematics (IK) solver and default IK tasks on DOF
    values, velocities, etc.
    """

    def __init__(self, path, root_body, free_flyer=True):
        super(IKRobot, self).__init__(path, root_body, free_flyer)
        self.ik = None  # created by self.init_ik()
        self.ik_lock = None
        self.ik_thread = None

    def init_ik(self, gains=None, weights=None, qd_lim=None):
        """
        Initialize the IK solver. Needs to be defined by child classes.

        INPUT:

        ``gains`` -- dictionary of default task gains
        ``weights`` -- dictionary of default task weights
        ``qd_lim`` -- maximum velocity (in [rad]), same for each joint
        """
        warn("no IK defined for robot of type %s" % (str(type(self))))

    def step_ik(self, dt):
        qd_active = self.ik.compute_velocity(dt)
        q_active = minimum(
            maximum(
                self.q_min_active,
                self.q_active + qd_active * dt),
            self.q_max_active)
        self.set_active_dof_values(q_active)
        self.set_active_dof_velocities(qd_active)

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
        q_active = self.q_active
        qd_active = zeros(self.nb_active_dofs)
        if debug:
            print "solve_ik(max_it=%d, conv_tol=%e)" % (max_it, conv_tol)
        for itnum in xrange(max_it):
            prev_cost = cur_cost
            cur_cost = self.ik.compute_cost(dt)
            cost_var = cur_cost - prev_cost
            if debug:
                print "%2d: %.3f (%+.2e)" % (itnum, cur_cost, cost_var)
            if abs(cost_var) < conv_tol:
                if abs(cur_cost) > 0.1:
                    warn("IK did not converge to solution. Is it feasible?"
                         "If so, try restarting from a random guess.")
                break
            qd_active = self.ik.compute_velocity(dt)
            q_active = minimum(
                maximum(self.q_min_active, q_active + qd_active * dt),
                self.q_max_active)
            self.set_active_dof_values(q_active)
        return itnum, cur_cost

    """
    Multi-threading
    """

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

    """
    Default tasks
    """

    def add_dof_task(self, dof_id, dof_ref, gain=None, weight=None):
        J = zeros(self.nb_dofs)
        J[dof_id] = 1.

        def residual(dt):
            return (dof_ref - self.q[dof_id])

        def jacobian():
            return J

        task_name = 'dof-%d' % dof_id
        self.ik.add_task(task_name, residual, jacobian, gain, weight)

    def add_min_acceleration_task(self, weight):
        """
        Minimize accelerations.

        INPUT:

        ``weight`` -- task weight

        .. NOTE::

            As the differential IK returns velocities, we approximate the task
            "minimize qdd" by "minimize (qd_next - qd)".
        """
        E = eye(self.nb_dofs)

        def residual(dt):
            return self.qd

        def jacobian():
            return E

        self.ik.add_task('qdd-min', residual, jacobian, weight=weight,
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
        E = eye(self.nb_dofs)

        def residual(dt):
            return -self.qd

        def jacobian():
            return E

        self.ik.add_task('qdmin', residual, jacobian, gain, weight)

    def add_posture_task(self, q_ref, exclude_dofs=None, gain=None,
                         weight=None):
        """
        Add a postural task, a common choice to regularize the weighted IK
        problem.

        INPUT:

        - ``exclude_dofs`` -- exclude some DOFs from the task
        - ``gain`` -- task gain
        - ``weight`` -- task weight

        .. NOTE::

            This is a regularization task, therefore ``weight`` should be low
            compared to the other task weights.
        """
        assert len(q_ref) == self.nb_dofs

        J_posture = eye(self.nb_dofs)
        if exclude_dofs is None:
            exclude_dofs = []
        if self.has_free_flyer:  # don't include translation coordinates
            exclude_dofs.extend([self.TRANS_X, self.TRANS_Y, self.TRANS_Z])
        for i in exclude_dofs:
            J_posture[i, i] = 0.

        def residual(dt):
            e = (q_ref - self.q)
            for j in exclude_dofs:
                e[j] = 0.
            return e

        def jacobian():
            return J_posture

        self.ik.add_task('posture', residual, jacobian, gain, weight)

    def remove_dof_task(self, dof_id):
        self.ik.remove_task('dof-%d' % dof_id)
