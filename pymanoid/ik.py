#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import dot, eye, hstack, maximum, minimum, ones, vstack, zeros
from threading import Lock
from time import time

from misc import norm
from optim import solve_qp
from sim import Process


class IKSolver(Process):

    """
    Compute velocities bringing the system closer to fulfilling a set of tasks.

    Parameters
    ----------
    robot : Robot
        Robot to be updated.
    active_dofs : list of integers, optional
        List of DOFs updated by the IK solver.
    doflim_gain : scalar, optional
        DOF-limit gain as described in [Kanoun12]_. In `this implementation
        <https://scaron.info/teaching/inverse-kinematics.html>`_, it should be
        between zero and one.

    Notes
    -----
    One unsatisfactory aspect of the DOF-limit gain is that it slows down the
    robot when approaching DOF limits. For instance, it may slow down a foot
    motion when approaching the knee singularity, despite the robot being able
    to move faster with a fully extended knee.
    """

    def __init__(self, robot, active_dofs=None, doflim_gain=0.5):
        super(IKSolver, self).__init__()
        if active_dofs is None:
            active_dofs = range(robot.nb_dofs)
        assert 0. <= doflim_gain <= 1.
        self.doflim_gain = doflim_gain
        self.qd = zeros(robot.nb_dofs)
        self.robot = robot
        self.tasks = {}
        self.tasks_lock = Lock()
        self.unsafe = False
        self.verbosity = 0
        #
        self.set_active_dofs(active_dofs)

    def set_active_dofs(self, active_dofs):
        nb_active_dofs = len(active_dofs)
        self.active_dofs = active_dofs
        self.nb_active_dofs = len(active_dofs)
        self.q_max = self.robot.q_max[active_dofs]
        self.q_min = self.robot.q_min[active_dofs]
        self.qd_max = +1. * ones(nb_active_dofs)
        self.qd_min = -1. * ones(nb_active_dofs)

    def add_task(self, task):
        """
        Add a new task to the IK solver.

        Parameters
        ----------
        task : Task
            New task to add to the list.

        Note
        ----
        This function is not made to be called frequently.
        """
        if task.name in self.tasks:
            raise Exception("Task '%s' already present in IK" % task.name)
        with self.tasks_lock:
            self.tasks[task.name] = task

    def clear_tasks(self):
        """Clear all tasks in the IK solver."""
        self.tasks = {}

    def __get_task_name(self, ident):
        name = ident if type(ident) is str else ident.name
        if not name.isupper():
            name = name.upper()
        return name

    def get_task(self, ident):
        """
        Get an active task from its name.

        Parameters
        ----------
        ident : string or object
            Name or object with a ``name`` field identifying the task.

        Returns
        -------
        task : Task or None
            The corresponding task if present, None otherwise.
        """
        name = self.__get_task_name(ident)
        with self.tasks_lock:
            if name not in self.tasks:
                return None
            return self.tasks[name]

    def remove_task(self, ident):
        """
        Remove a task.

        Parameters
        ----------
        ident : string or object
            Name or object with a ``name`` field identifying the task.
        """
        name = self.__get_task_name(ident)
        with self.tasks_lock:
            if name not in self.tasks:
                return
            del self.tasks[name]

    def update_task(self, ident, task):
        """
        Update a task.

        Parameters
        ----------
        ident : string or object
            Name or object with a ``name`` field identifying the task.
        """
        name = self.__get_task_name(ident)
        assert task.name == name
        self.remove_task(name)
        self.add_task(task)

    def compute_cost(self, dt):
        """
        Compute the IK cost of the present system state for a time step of dt.

        Parameters
        ----------
        dt : scalar
            Time step in [s].
        """
        return sum(task.cost(dt) for task in self.tasks.itervalues())

    def __compute_qp_common(self, dt):
        n = self.nb_active_dofs
        q = self.robot.q[self.active_dofs]
        P = zeros((n, n))
        v = zeros(n)
        with self.tasks_lock:
            for task in self.tasks.itervalues():
                J = task.jacobian()[:, self.active_dofs]
                r = task.residual(dt)
                P += task.weight * dot(J.T, J)
                v += task.weight * dot(-r.T, J)
        qd_max_doflim = self.doflim_gain * (self.q_max - q) / dt
        qd_min_doflim = self.doflim_gain * (self.q_min - q) / dt
        qd_max = minimum(self.qd_max, qd_max_doflim)
        qd_min = maximum(self.qd_min, qd_min_doflim)
        return (P, v, qd_max, qd_min)

    def __solve_qp(self, P, v, G, h):
        try:
            qd_active = solve_qp(P, v, G, h)
            self.qd[self.active_dofs] = qd_active
        except ValueError as e:
            if "matrix G is not positive definite" in e:
                raise Exception(
                    "rank deficiency in IK problem, "
                    "did you add a regularization task?")
            raise
        return self.qd

    def compute_velocity_fast(self, dt):
        """
        Compute a new velocity satisfying all tasks at best.

        Parameters
        ----------
        dt : scalar
            Time step in [s].

        Returns
        -------
        qd : array
            Active joint velocity vector.

        Note
        ----
        This QP formulation is the default for
        :func:`pymanoid.ik.IKSolver.solve` (posture generation) as it converges
        faster.

        Notes
        -----
        The method implemented in this function is reasonably fast but may
        become unstable when some tasks are widely infeasible and the optimum
        saturates joint limits. In such situations, it is better to use
        :func:`pymanoid.ik.IKSolver.compute_velocity_safe`.

        The returned velocity minimizes squared residuals as in the weighted
        cost function, which corresponds to the Gauss-Newton algorithm. Indeed,
        expanding the square expression in ``cost(task, qd)`` yields

        .. math::

            \\mathrm{minimize} \\quad
                \\dot{q} J^T J \\dot{q} - 2 r^T J \\dot{q}

        Differentiating with respect to :math:`\\dot{q}` shows that the minimum
        is attained for :math:`J^T J \\dot{q} = r`, where we recognize the
        Gauss-Newton update rule.
        """
        n = self.nb_active_dofs
        P, v, qd_max, qd_min = self.__compute_qp_common(dt)
        G = vstack([+eye(n), -eye(n)])
        h = hstack([qd_max, -qd_min])
        return self.__solve_qp(P, v, G, h)

    def compute_velocity_safe(self, dt, margin_reg=1e-5, margin_lin=1e-3):
        """
        Compute a new velocity satisfying all tasks at best, while trying to
        stay away from kinematic constraints.

        Parameters
        ----------
        dt : scalar
            Time step in [s].
        margin_reg : scalar
            Regularization term on margin variables.
        margin_lin : scalar
            Linear penalty term on margin variables.

        Returns
        -------
        qd : array
            Active joint velocity vector.

        Note
        ----
        This QP formulation is the default for :func:`pymanoid.ik.IKSolver.step`
        as it has a more numerically-stable behavior.

        Notes
        -----
        This is a variation of the QP from
        :func:`pymanoid.ik.IKSolver.compute_velocity_fast` that was reported in
        Equation (10) of [Nozawa16]_. DOF limits are better taken care of by
        margin variables, but the variable count doubles and the QP takes
        roughly 50% more time to solve.
        """
        n = self.nb_active_dofs
        E, Z = eye(n), zeros((n, n))
        P0, v0, qd_max, qd_min = self.__compute_qp_common(dt)
        P = vstack([hstack([P0, Z]), hstack([Z, margin_reg * E])])
        v = hstack([v0, -margin_lin * ones(n)])
        G = vstack([
            hstack([+E, +E / dt]), hstack([-E, +E / dt]), hstack([Z, -E])])
        h = hstack([qd_max, -qd_min, zeros(n)])
        return self.__solve_qp(P, v, G, h)

    def step(self, dt, unsafe=False):
        """
        Apply velocities computed by inverse kinematics.

        Parameters
        ----------
        dt : scalar
            Time step in [s].
        unsafe : bool, optional
            When set, use the faster but less numerically-stable method
            implemented in :func:`pymanoid.ik.IKSolver.compute_velocity_fast`.
        """
        q = self.robot.q
        if unsafe or self.unsafe:
            qd = self.compute_velocity_fast(dt)
        else:  # safe formulation is the default
            qd = self.compute_velocity_safe(dt)
        if self.verbosity >= 3:
            print "\n                TASK      COST",
            print "\n------------------------------"
            for task in self.tasks.itervalues():
                J = task.jacobian()
                r = task.residual(dt)
                print "%20s  %.2e" % (task.name, norm(dot(J, qd) - r))
            print ""
        self.robot.set_dof_values(q + qd * dt, clamp=True)
        self.robot.set_dof_velocities(qd)

    def solve(self, max_it=1000, cost_stop=1e-10, impr_stop=1e-5, dt=5e-3):
        """
        Compute joint-angles that satisfy all kinematic constraints at best.

        Parameters
        ----------
        max_it : integer
            Maximum number of solver iterations.
        cost_stop : scalar
            Stop when cost value is below this threshold.
        conv_tol : scalar, optional
            Stop when cost improvement (relative variation from one iteration to
            the next) is less than this threshold.
        dt : scalar, optional
            Time step in [s].

        Returns
        -------
        itnum : int
            Number of solver iterations.
        cost : scalar
            Final value of the cost function.

        Note
        ----
        Good values of `dt` depend on the weights of the IK tasks. Small values
        make convergence slower, while big values make the optimization unstable
        (in which case there may be no convergence at all).
        """
        t0 = time()
        if self.verbosity >= 2:
            print "Solving IK with max_it=%d, conv_stop=%e, impr_stop=%e" % (
                max_it, cost_stop, impr_stop)
        cost = 100000.
        self.qd_max *= 1000
        self.qd_min *= 1000
        for itnum in xrange(max_it):
            prev_cost = cost
            cost = self.compute_cost(dt)
            impr = abs(cost - prev_cost) / prev_cost
            if self.verbosity >= 2:
                print "%2d: %.3e (impr: %+.2e)" % (itnum, cost, impr)
            if abs(cost) < cost_stop or impr < impr_stop:
                break
            self.step(dt, unsafe=True)
        self.robot.set_dof_velocities(zeros(self.robot.qd.shape))
        self.qd_max /= 1000
        self.qd_min /= 1000
        if self.verbosity >= 1:
            print "IK solved in %d iterations (%.1f ms) with cost %e" % (
                itnum, 1000 * (time() - t0), cost)
        return itnum, cost

    def on_tick(self, sim):
        self.step(sim.dt)
