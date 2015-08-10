#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of openravepypy.
#
# openravepypy is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# openravepypy is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openravepypy. If not, see <http://www.gnu.org/licenses/>.


import time

from numpy import minimum, maximum, zeros
from numpy import dot, eye, ones, vstack, hstack
from toolbox import cvxopt_solve_qp


class DiffIKSolver(object):

    class Task(object):

        def __init__(self, err_fun, jacobian_fun, gain):
            self.err = err_fun
            self.gain = gain
            self.jacobian = jacobian_fun

        def vel(self, q):
            return self.gain * self.err(q)

    class Objective(object):

        def __init__(self, task, weight):
            self.task = task
            self.weight = weight

        def err(self, q):
            return self.task.err(q)

        def jacobian(self, q):
            return self.task.jacobian(q)

        def vel(self, q):
            return self.task.vel(q)

    def __init__(self, robot, active_dofs, K_doflim, reg_weight, vel_lim,
                 conv_tol, dof_lim_scale=0.95):
        n = len(active_dofs)
        q_avg = .5 * (robot.q_max + robot.q_min)
        q_dev = .5 * (robot.q_max - robot.q_min)

        self.active_dofs = active_dofs
        self.constraints = []
        self.conv_tol = conv_tol
        self.I = eye(n)
        self.K_doflim = K_doflim
        self.n = n
        self.objectives = []
        self.q_max = (q_avg + dof_lim_scale * q_dev)[active_dofs]
        self.q_min = (q_avg - dof_lim_scale * q_dev)[active_dofs]
        self.qd_max = (+vel_lim * ones(robot.nb_dof))[active_dofs]
        self.qd_min = (-vel_lim * ones(robot.nb_dof))[active_dofs]
        self.reg_weight = reg_weight
        self.robot = robot

    def reset(self):
        self.constraints = []
        self.objectives = []

    @property
    def q(self):
        return self.robot.get_dof_values(self.active_dofs)

    def add_constraint(self, err_fun, jacobian_fun, gain):
        self.constraints.append(self.Task(err_fun, jacobian_fun, gain))

    def add_objective(self, err_fun, jacobian_fun, gain, weight):
        task = self.Task(err_fun, jacobian_fun, gain)
        self.objectives.append(self.Objective(task, weight))

    def compute_instant_vel(self):
        qd_max = minimum(self.qd_max, self.K_doflim * (self.q_max - self.q))
        qd_min = maximum(self.qd_min, self.K_doflim * (self.q_min - self.q))
        J_list = [c.jacobian(self.q) for c in self.constraints]
        v_list = [c.vel(self.q) for c in self.constraints]
        P = self.reg_weight * self.I
        q = zeros(len(self.q))

        for obj in self.objectives:
            v, J = obj.vel(self.q), obj.jacobian(self.q)
            P += obj.weight * dot(J.T, J)
            q += obj.weight * dot(-v.T, J)

        G = vstack([+self.I, -self.I])
        h = hstack([qd_max, -qd_min])
        if J_list:
            A = vstack(J_list)
            b = hstack(v_list)
            return cvxopt_solve_qp(P, q, G, h, A, b)
        return cvxopt_solve_qp(P, q, G, h)

    def add_link_objective(self, link, target_pose, gain=1., weight=1.):
        def err(q):
            cur_pose = self.robot.compute_link_pose(link, q, self.active_dofs)
            return target_pose - cur_pose

        def J(q):
            pose = self.robot.compute_link_pose(link, q, self.active_dofs)
            J = self.robot.compute_link_pose_jacobian(link, q, self.active_dofs)
            if pose[0] < 0:  # convention: cos(alpha) > 0
                J[:4, :] *= -1
            return J

        self.add_objective(err, J, gain=gain, weight=weight)

    def add_com_objective(self, target_com, gain=0.5, weight=0.1):
        def err(q):
            cur_com = self.robot.compute_com(q, self.active_dofs)
            return target_com - cur_com

        def J(q):
            return self.robot.compute_com_jacobian(
                q, dof_indices=self.active_dofs)

        self.add_objective(err, J, gain=gain, weight=weight)

    def add_dof_objective(self, dof_id, target_angle, gain=.1, weight=0.0005):
        def err(q):
            return target_angle - q[dof_id]

        def J(q):
            j = zeros((len(q),))
            j[dof_id] = 1.
            return j

        self.add_objective(err, J, gain=gain, weight=weight)


class IKTracker(DiffIKSolver):

    def track(self, dt, callback=None, duration=120.):
        t = 0.
        q = self.robot.get_dof_values(self.active_dofs)
        while t < duration:
            t0 = time.time()
            qd = self.compute_instant_vel()
            if callback is not None:
                callback()
            q = q + qd * dt
            q = minimum(maximum(q, self.q_min), self.q_max)
            t = t + dt
            self.robot.set_dof_values(q, self.active_dofs)  # needed by vel()
            Dt = dt - time.time() + t0
            if Dt > 0:
                time.sleep(Dt)


class IKSolver(DiffIKSolver):

    def compute_objective(self):
        def sq(v): return dot(v, v)
        return sum(obj.weight * sq(obj.err(self.q)) for obj in self.objectives)

    def solve(self, dt=.1, max_it=100, debug=False):
        cur_obj = 1000
        for itnum in xrange(max_it):
            prev_obj = cur_obj
            cur_obj = self.compute_objective()
            if abs(cur_obj - prev_obj) < self.conv_tol:
                break
            qd = self.compute_instant_vel()
            q_next = self.q + qd * dt
            q_next = minimum(maximum(q_next, self.q_min), self.q_max)
            self.robot.set_dof_values(q_next, self.active_dofs)
            if debug:
                print "%2d:" % itnum, cur_obj
        return itnum, cur_obj
