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
from qp import solve_qp
from threading import Lock
from warnings import warn


class DiffIKSolver(object):

    """
    The differential IK solver computes velocities ``qd`` that bring the system
    closer to fulfilling a set of tasks. A task is defined by

    - ``residual(dt)``, specifying the (desired - current) workspace
      displacement
    - ``jacobian()``, mapping joint velocities to workspace displacements
    - two scalars ``gain`` and ``weight``.

    A task is perfectly achieved when:

        jacobian() * qd == gain * residual(q, qd, dt) / dt     (1)

    To tend toward this, each task adds a term

        cost(task, qd) = weight * |jacobian * qd - gain * residual / dt|^2

    to the cost function of the optimization problem solved at each time step
    by the differential IK:

        minimize    sum_tasks cost(task, qd)
            s.t.    qd_min <= qd <= qd_max

    .. NOTE::

        Minimizing squared residuals as in the cost functions above corresponds
        to the Gauss-Newton algorithm
        <https://en.wikipedia.org/wiki/Gauss%E2%80%93Newton_algorithm>. Indeed,
        expanding the square expression in cost(task, qd) yields

            minimize    qd * (J.T * J) * qd - 2 * (residual / dt) * J * qd

        Differentiating with respect to ``qd`` shows that the minimum is
        attained for (J.T * J) * qd == (residual / dt), and we recognize the
        Gauss-Newton update rule.
    """

    def __init__(self, robot, default_gains=None, default_weights=None,
                 doflim_gain=1.):
        """
        Initialize the differential IK solver.

        INPUT:

        - ``robot`` -- upper DOF limit
        - ``gains`` -- dictionary of default task gains
        - ``weights`` -- dictionary of default task weights
        - ``doflim`` -- (optional, default: 0.5) a special gain converting joint
                     limits into a suitable velocity bound. See Equation (50) in
                     <http://www.roboticsproceedings.org/rss07/p21.pdf>.
        """
        self.default_gains = {}
        self.default_weights = {}
        self.doflim_gain = doflim_gain
        self.gains = {}
        self.jacobians = {}
        self.residuals = {}
        self.robot = robot
        self.tasks = {}
        self.tasks_lock = Lock()
        self.weights = {}
        if default_gains is not None:
            self.default_gains.update(default_gains)
        if default_weights is not None:
            self.default_weights.update(default_weights)

    def add_task(self, name, error, jacobian, gain=None, weight=None,
                 task_type=None, unit_gain=False):
        """
        Add a new task in the IK.

        INPUT:

        ``name`` -- task name, used as identifier for e.g. removal
        ``error`` -- error function of the task (depends on q, qd and dt)
        ``jacobian`` -- jacobian function of the task (depends on q only)
        ``gain`` -- task gain
        ``weight`` -- task weight
        ``task_type`` -- for some tasks such as contact, ``name`` corresponds to
                         a robot link name; ``task_type`` is then used to fetch
                         default gain and weight values
        ``unit_gain`` -- some tasks have a different formulation where the gain
                         is one and there is no division by dt in Equation (1);
                         set ``unit_gain=1`` for this behavior

        .. NOTE::

            This function is not made to be called frequently.

        """
        if name in self.tasks:
            raise Exception("Task '%s' already present in IK" % name)
        with self.tasks_lock:
            self.tasks[name] = True
            self.residuals[name] = error
            self.jacobians[name] = jacobian
            if unit_gain:
                if name in self.gains:
                    del self.gains[name]
            elif gain is not None:
                self.gains[name] = gain
            elif name in self.default_gains:
                self.gains[name] = self.default_gains[name]
            elif task_type in self.default_gains:
                self.gains[name] = self.default_gains[task_type]
            else:
                msg = "No gain provided for task '%s'" % name
                if task_type is not None:
                    msg += " (task_type='%s')" % task_type
                raise Exception(msg)
            if weight is not None:
                self.weights[name] = weight
            elif name in self.default_weights:
                self.weights[name] = self.default_weights[name]
            elif task_type in self.default_weights:
                self.weights[name] = self.default_weights[task_type]
            else:
                msg = "No weight provided for task '%s'" % name
                if task_type is not None:
                    msg += " (task_type='%s')" % task_type
                raise Exception(msg)
            assert unit_gain or self.gains[name] < 1. + 1e-10, \
                "Task gains should be between 0. and 1 (%f)." % self.gains[name]
            assert self.weights[name] > 0., \
                "Task weights should be positive"

    def remove_task(self, name):
        with self.tasks_lock:
            if name not in self.tasks:
                warn("no task '%s' to remove" % name)
                return
            del self.tasks[name]

    def compute_cost(self, dt):
        def sq(e):
            return dot(e, e)

        def cost(task):
            return self.weights[task] * sq(self.residuals[task](dt))

        return sum(cost(task) for task in self.tasks)

    def compute_velocity(self, dt):
        """
        Compute a new velocity satisfying all tasks at best, while staying
        within joint-velocity limits.

        INPUT:

        ``q`` -- current joint vector
        ``qd`` -- current joint velocities
        ``dt`` -- time step until next call to the IK

        .. NOTE::

        """
        n = self.robot.nb_active_dofs
        q = self.robot.q_active
        q_max = self.robot.q_max[self.robot.active_dofs]
        q_min = self.robot.q_min[self.robot.active_dofs]
        qd_max = self.robot.qd_max[self.robot.active_dofs]
        qd_min = self.robot.qd_min[self.robot.active_dofs]
        E = eye(n)
        P = zeros((n, n))
        r = zeros(n)
        with self.tasks_lock:
            for task in self.tasks:
                J = self.jacobians[task]()[:, self.robot.active_dofs]
                if task in self.gains:
                    e = self.gains[task] * (self.residuals[task](dt) / dt)
                else:  # for tasks added with unit_gain=True
                    e = self.residuals[task](dt)
                P += self.weights[task] * dot(J.T, J)
                r += self.weights[task] * dot(-e.T, J)
        qd_max_doflim = self.doflim_gain * (q_max - q) / dt
        qd_min_doflim = self.doflim_gain * (q_min - q) / dt
        qd_max = minimum(qd_max, qd_max_doflim)
        qd_min = maximum(qd_min, qd_min_doflim)
        G = vstack([+E, -E])
        h = hstack([qd_max, -qd_min])
        return solve_qp(P, r, G, h)
