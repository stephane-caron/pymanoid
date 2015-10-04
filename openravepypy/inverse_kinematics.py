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


from numpy import dot, eye, hstack, maximum, minimum, ones, vstack, zeros
from toolbox import cvxopt_solve_qp


class DiffIKSolver(object):

    class Task(object):

        def __init__(self, error_fun, jacobian_fun, gain):
            self.error = error_fun
            self.gain = gain
            self.jacobian = jacobian_fun

        def velocity(self, q, qd):
            return self.gain * self.error(q, qd)

    class Objective(object):

        def __init__(self, task, weight):
            self.task = task
            self.weight = weight

        def error(self, q, qd):
            return self.task.error(q, qd)

        def sq_error(self, q, qd):
            e = self.error(q, qd)
            return dot(e, e)

        def jacobian(self, q):
            return self.task.jacobian(q)

        def velocity(self, q, qd):
            return self.task.velocity(q, qd)

    def __init__(self, robot, dt, qd_lim, K_doflim, doflim_scale):
        n = len(robot.q)
        self.I = eye(n)
        self.K_doflim = K_doflim
        self.constraints = []
        self.dt = dt
        self.objectives = []
        self.q_max = robot.q_max
        self.q_min = robot.q_min
        self.qd_max = +qd_lim * ones(n)
        self.qd_min = -qd_lim * ones(n)

    def identity(self, q):
        """Default jacobian for joint-angle objectives."""
        return self.I

    def add_constraint(self, error_fun, jacobian_fun, gain):
        self.constraints.append(self.Task(error_fun, jacobian_fun, gain))

    def add_objective(self, error_fun, jacobian_fun, gain, weight):
        task = self.Task(error_fun, jacobian_fun, gain)
        self.objectives.append(self.Objective(task, weight))

    def compute_objective(self, q, qd):
        return sum(obj.weight * obj.sq_error(q, qd) for obj in self.objectives)

    def compute_velocity(self, q, qd):
        P = zeros(self.I.shape)
        r = zeros(self.q_max.shape)
        for obj in self.objectives:
            J = obj.jacobian(q)
            P += obj.weight * dot(J.T, J)
            r += obj.weight * dot(-obj.velocity(q, qd).T, J)
        if self.K_doflim is not None:
            qd_max = minimum(self.qd_max, self.K_doflim * (self.q_max - q))
            qd_min = maximum(self.qd_min, self.K_doflim * (self.q_min - q))
        else:
            qd_max = self.qd_max
            qd_min = self.qd_min
        G = vstack([+self.I, -self.I])
        h = hstack([qd_max, -qd_min])
        if self.constraints:
            A = vstack([c.jacobian() for c in self.constraints])
            b = hstack([c.velocity() for c in self.constraints])
            return cvxopt_solve_qp(P, r, G, h, A, b)
        return cvxopt_solve_qp(P, r, G, h)
