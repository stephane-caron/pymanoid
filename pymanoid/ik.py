#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
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
# You should have received a copy of the GNU Lesser General Public License along
# with pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import dot, eye, hstack, maximum, minimum, ones, vstack, zeros
from threading import Lock

from misc import norm, warn
from optim import solve_qp
from sim import Process
from tasks import ContactTask, DOFTask, PoseTask


RANK_DEFICIENCY_MSG = "rank deficiency in IK problem, " \
    "did you add a regularization task?"


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

    __DEFAULT_GAINS = {
        'COM': 0.85,
        'CONTACT': 0.85,
        'DOF': 0.85,
        'MIN_ACCEL': 0.85,
        'MIN_CAM': 0.85,
        'MIN_VEL': 0.85,
        'PENDULUM': 0.85,
        'POSE': 0.85,
        'POSTURE': 0.85,
    }

    __DEFAULT_WEIGHTS = {
        'CONTACT': 1.,
        'COM': 1e-2,
        'POSE': 1e-3,
        'MIN_ACCEL': 1e-4,
        'MIN_CAM': 1e-4,
        'DOF': 1e-5,
        'POSTURE': 1e-6,
    }

    def __init__(self, robot, active_dofs=None, doflim_gain=0.5):
        super(IKSolver, self).__init__()
        if active_dofs is None:
            active_dofs = range(robot.nb_dofs)
        assert 0. <= doflim_gain <= 1.
        self.default_gains = {}
        self.default_weights = {}
        self.doflim_gain = doflim_gain
        self.qd = zeros(robot.nb_dofs)
        self.robot = robot
        self.tasks = {}
        self.tasks_lock = Lock()
        self.unsafe = False
        #
        self.set_active_dofs(active_dofs)
        self.set_default_gains()
        self.set_default_weights()

    def set_active_dofs(self, active_dofs):
        """
        Set DOF indices modified by the IK.

        Parameters
        ----------
        active_dofs : list of integers
            List of DOF indices.
        """
        self.active_dofs = active_dofs
        self.nb_active_dofs = len(active_dofs)
        self.q_max = self.robot.q_max[active_dofs]
        self.q_min = self.robot.q_min[active_dofs]
        self.qd_max = +1. * self.robot.qd_lim[active_dofs]
        self.qd_min = -1. * self.robot.qd_lim[active_dofs]

    def set_default_gains(self, default_gains=None):
        """
        Set default gains for new tasks.

        Parameters
        ----------
        default_gains : string -> int dictionary, optional
            Dictionary mapping task labels to default gain values.

        Note
        ----
        When called with no argument, this function will use a set of "sane"
        default values.
        """
        if default_gains is None:  # sane defaults
            default_gains = self.__DEFAULT_GAINS
        self.default_gains.update(default_gains)

    def set_default_weights(self, default_weights=None):
        """
        Set default cost-function weights for new tasks.

        Parameters
        ----------
        default_weights : string -> double dictionary, optional
            Dictionary mapping task labels to default weight values.

        Note
        ----
        When called with no argument, this function will use a set of "sane"
        default values.
        """
        if default_weights is None:  # sane defaults
            default_weights = self.__DEFAULT_WEIGHTS
        else:  # default_weights is not None
            for key in default_weights:
                if key not in self.__DEFAULT_WEIGHTS:
                    warn("unknown key '%s' for IK default weights" % key)
        self.default_weights.update(default_weights)

    def set_task_weights(self, weights):
        """
        Set cost-function weights for existing tasks.

        Parameters
        ----------
        weights : string -> double dictionary
            Dictionary mapping task labels to default weight values.
        """
        for (name, weight) in weights.iteritems():
            self.tasks[name].weight = weight

    def __fill_task_gain(self, task):
        if task.name in self.default_gains:
            task.gain = self.default_gains[task.name]
        elif type(task) is ContactTask \
                and 'CONTACT' in self.default_gains:
            task.gain = self.default_gains['CONTACT']
        elif type(task) is DOFTask and 'DOF' in self.default_gains:
            task.gain = self.default_gains['DOF']
        elif type(task) is PoseTask and 'POSE' in self.default_gains:
            task.gain = self.default_gains['POSE']

    def __fill_task_weight(self, task):
        if task.name in self.default_weights:
            task.weight = self.default_weights[task.name]
        elif type(task) is ContactTask \
                and 'CONTACT' in self.default_weights:
            task.weight = self.default_weights['CONTACT']
        elif type(task) is DOFTask and 'DOF' in self.default_weights:
            task.weight = self.default_weights['DOF']
        elif type(task) is PoseTask and 'POSE' in self.default_weights:
            task.weight = self.default_weights['POSE']

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
        if task.weight is None:
            self.__fill_task_weight(task)
            if task.weight is None:
                raise Exception("no weight provided for task '%s'" % task.name)
        if task.gain is None:
            self.__fill_task_gain(task)
            if task.gain is None:
                raise Exception("no gain provided for task '%s'" % task.name)
        with self.tasks_lock:
            self.tasks[task.name] = task

    def clear_tasks(self):
        """Clear all tasks in the IK solver."""
        self.tasks = {}

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
        name = ident if type(ident) is str else ident.name
        with self.tasks_lock:
            if name not in self.tasks:
                return None
            return self.tasks[name]

    def print_task_costs(self, qd, dt):
        """
        Print task costs for the current IK step.

        Parameters
        ----------
        qd : array
            Robot DOF velocities.
        dt : scalar
            Timestep for the IK.
        """
        print "\n                TASK      COST",
        print "\n------------------------------"
        for task in self.tasks.itervalues():
            J = task.jacobian()
            r = task.residual(dt)
            print "%20s  %.2e" % (task.name, norm(dot(J, qd) - r))
        print ""

    def remove_task(self, ident):
        """
        Remove a task.

        Parameters
        ----------
        ident : string or object
            Name or object with a ``name`` field identifying the task.
        """
        name = ident if type(ident) is str else ident.name
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
        name = ident if type(ident) is str else ident.name
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

    def build_qp_matrices(self, dt):
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
            Vector of active joint velocities.

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
        P, v, qd_max, qd_min = self.build_qp_matrices(dt)
        G = vstack([+eye(n), -eye(n)])
        h = hstack([qd_max, -qd_min])
        try:
            x = solve_qp(P, v, G, h)
            self.qd[self.active_dofs] = x
        except ValueError as e:
            if "matrix G is not positive definite" in e:
                raise Exception(RANK_DEFICIENCY_MSG)
            raise
        return self.qd

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
            Vector of active joint velocities.

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
        P0, v0, qd_max, qd_min = self.build_qp_matrices(dt)
        P = vstack([hstack([P0, Z]), hstack([Z, margin_reg * E])])
        v = hstack([v0, -margin_lin * ones(n)])
        G = vstack([
            hstack([+E, +E / dt]), hstack([-E, +E / dt]), hstack([Z, -E])])
        h = hstack([qd_max, -qd_min, zeros(n)])
        try:
            x = solve_qp(P, v, G, h)
            self.qd[self.active_dofs] = x[:n]
        except ValueError as e:
            if "matrix G is not positive definite" in e:
                raise Exception(RANK_DEFICIENCY_MSG)
            raise
        return self.qd

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
        self.robot.set_dof_values(q + qd * dt, clamp=True)
        self.robot.set_dof_velocities(qd)

    def solve(self, max_it=1000, cost_stop=1e-10, impr_stop=1e-5, dt=5e-3,
              qd_relax_fact=10., qd_relax_steps=2, debug=False):
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
        qd_relax_fact : scalar, optional
            Relaxation factor on DOF velocity limits.
        qd_relax_steps : int, optional
            Number of DOF-velocity relaxation stages.
        debug : bool, optional
            Set to True for additional debug messages.

        Returns
        -------
        nb_it : int
            Number of solver iterations.
        cost : scalar
            Final value of the cost function.

        Notes
        -----
        Good values of `dt` depend on the weights of the IK tasks. Small values
        make convergence slower, while big values make the optimization unstable
        (in which case there may be no convergence at all).

        To speed up convergence, this function will relax DOF velocity limits at
        first, then progressively restore them. This behavior is set by the two
        parameters `qd_relax_fact` (relaxation factor) and `qd_relax_steps`
        (number of relaxation stages).
        """
        cost = 100000.
        self.qd_max *= qd_relax_fact ** qd_relax_steps
        self.qd_min *= qd_relax_fact ** qd_relax_steps
        N = qd_relax_steps + 1
        qd_stepdowns = [max_it * i / N for i in xrange(1, N)]
        for itnum in xrange(max_it):
            if itnum in qd_stepdowns:
                self.qd_max /= qd_relax_fact
                self.qd_min /= qd_relax_fact
            prev_cost = cost
            cost = self.compute_cost(dt)
            impr = abs(cost - prev_cost) / prev_cost
            if debug:
                print "%2d: %.3e (impr: %+.2e)" % (itnum, cost, impr)
            if abs(cost) < cost_stop or impr < impr_stop:
                break
            self.step(dt, unsafe=(itnum < max_it / 2))
        self.qd_max = +1. * self.robot.qd_lim[self.active_dofs]
        self.qd_min = -1. * self.robot.qd_lim[self.active_dofs]
        self.robot.set_dof_velocities(zeros(self.robot.qd.shape))
        return 1 + itnum, cost

    def on_tick(self, sim):
        """
        Step the IK at each simulation tick.

        Parameters
        ----------
        sim : Simulation
            Simulation instance.
        """
        self.step(sim.dt)
