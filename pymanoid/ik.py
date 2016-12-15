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

from numpy import dot, eye, hstack, maximum, minimum, vstack, zeros
from optim import solve_qp
from threading import Lock
from warnings import warn


class IKError(Exception):

    pass


class VelocitySolver(object):

    """
    Compute velocities bringing the system closer to fulfilling a set of tasks.

    See  for details.
    """

    def __init__(self, robot, active_dofs, doflim_gain):
        """
        Initialize the solver.

        INPUT:

        - ``robot`` -- upper DOF limit
        - ``active_dofs`` -- list of DOFs used by the IK
        - ``doflim_gain`` -- gain between 0 and 1 used for DOF limits

        The ``doflim_gain`` is described in [Kanoun2012]. In this
        implementation, it should be between 0. and 1. [Caron2016]. One
        unsatisfactory aspect of this solution is that it artificially slows
        down the robot when approaching DOF limits. For instance, it may slow
        down a foot motion when approaching the knee singularity, despite the
        robot being able to move faster with a fully extended knee.

        REFERENCES:

        .. [Caron2016] <https://scaron.info/teaching/inverse-kinematics.html>
        .. [Kanoun2012] <http://www.roboticsproceedings.org/rss07/p21.pdf>
        """
        assert 0. <= doflim_gain <= 1.
        nb_active_dofs = len(active_dofs)
        qp_G = vstack([+eye(nb_active_dofs), -eye(nb_active_dofs)])
        self.active_dofs = active_dofs
        self.doflim_gain = doflim_gain
        self.nb_active_dofs = len(active_dofs)
        self.q_max = robot.q_max[active_dofs]
        self.q_min = robot.q_min[active_dofs]
        self.qd = zeros(robot.nb_dofs)
        self.qd_max = robot.qd_max[active_dofs]
        self.qd_min = robot.qd_min[active_dofs]
        self.qp_G = qp_G
        self.robot = robot
        self.tasks = {}
        self.tasks_lock = Lock()

    def add_task(self, task):
        """
        Add a new task in the IK.

        INPUT:

        - ``task`` -- Task object

        .. NOTE::

            This function is not made to be called frequently.
        """
        task.check()
        if task.name in self.tasks:
            raise Exception("Task '%s' already present in IK" % task.name)
        with self.tasks_lock:
            self.tasks[task.name] = task

    def add_tasks(self, tasks):
        for task in tasks:
            self.add_task(task)

    def get_task(self, name):
        """
        Get an active task from its name.

        INPUT:

        - ``name`` -- task name
        """
        with self.tasks_lock:
            if name not in self.tasks:
                warn("no task with name '%s'" % name)
                return
            return self.tasks[name]

    def remove_task(self, name):
        """
        Remove a task from the IK.

        INPUT:

        - ``name`` -- task name
        """
        with self.tasks_lock:
            if name not in self.tasks:
                warn("no task '%s' to remove" % name)
                return
            del self.tasks[name]

    def compute_cost(self, dt):
        return sum(task.cost(dt) for task in self.tasks.itervalues())

    def compute_velocity(self, dt):
        """
        Compute a new velocity satisfying all tasks at best, while staying
        within joint-velocity limits.

        INPUT:

        - ``dt`` -- time step

        .. NOTE::

            Minimizing squared residuals as in the weighted cost function
            corresponds to the Gauss-Newton algorithm
            <https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>.
            Indeed, expanding the square expression in cost(task, qd) yields

                minimize    qd * (J.T * J) * qd - 2 * (residual / dt) * J * qd

            Differentiating with respect to ``qd`` shows that the minimum is
            attained for (J.T * J) * qd == (residual / dt), and we recognize the
            Gauss-Newton update rule.
        """
        n = self.nb_active_dofs
        qp_P = zeros((n, n))
        qp_q = zeros(n)
        with self.tasks_lock:
            for task in self.tasks.itervalues():
                J = task.jacobian()[:, self.active_dofs]
                r = task.residual(dt)
                qp_P += task.weight * dot(J.T, J)
                qp_q += task.weight * dot(-r.T, J)
        q = self.robot.q[self.active_dofs]
        qd_max_doflim = self.doflim_gain * (self.q_max - q) / dt
        qd_min_doflim = self.doflim_gain * (self.q_min - q) / dt
        qd_max = minimum(self.qd_max, qd_max_doflim)
        qd_min = maximum(self.qd_min, qd_min_doflim)
        # qp_G = vstack([+eye(n), -eye(n)])
        qp_G = self.qp_G  # saved to avoid recomputations
        qp_h = hstack([qd_max, -qd_min])
        try:
            qd_active = solve_qp(qp_P, qp_q, qp_G, qp_h)
            self.qd[self.active_dofs] = qd_active
        except ValueError as e:
            if "matrix G is not positive definite" in e:
                msg = "rank deficiency. Did you add a regularization task?"
                raise IKError(msg)
            raise
        return self.qd
