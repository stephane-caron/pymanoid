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


from numpy import minimum, maximum, zeros
from numpy import dot, eye, vstack, hstack
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

        def sq_err(self, q):
            e = self.err(q)
            return dot(e, e)

        def jacobian(self, q):
            return self.task.jacobian(q)

        def vel(self, q):
            return self.task.vel(q)

    def __init__(self, q_lim, qd_lim, K_doflim, reg_weight):
        q_min, q_max = q_lim
        qd_min, qd_max = qd_lim
        n = len(q_min)
        self.I = eye(n)
        self.K_doflim = K_doflim
        self.constraints = []
        self.n = n
        self.objectives = []
        self.q_min = q_min
        self.q_max = q_max
        self.qd_min = qd_min
        self.qd_max = qd_max
        self.reg_weight = reg_weight

    def add_constraint(self, err_fun, jacobian_fun, gain):
        self.constraints.append(self.Task(err_fun, jacobian_fun, gain))

    def add_objective(self, err_fun, jacobian_fun, gain, weight):
        task = self.Task(err_fun, jacobian_fun, gain)
        self.objectives.append(self.Objective(task, weight))

    def compute_objective(self, q):
        return sum(obj.weight * obj.sq_err(q) for obj in self.objectives)

    def compute_delta(self, q):
        P = self.reg_weight * self.I
        r = zeros(self.n)
        for obj in self.objectives:
            J = obj.jacobian(q)
            P += obj.weight * dot(J.T, J)
            r += obj.weight * dot(-obj.vel(q).T, J)
        if self.K_doflim is not None:
            qd_max = minimum(self.qd_max, self.K_doflim * (self.q_max - q))
            qd_min = maximum(self.qd_min, self.K_doflim * (self.q_min - q))
        else:
            qd_max = self.qd_max
            qd_min = self.qd_min
        G = vstack([+self.I, -self.I])
        h = hstack([qd_max, -qd_min])
        if self.constraints:
            A = vstack([c.jacobian(q) for c in self.constraints])
            b = hstack([c.vel(q) for c in self.constraints])
            return cvxopt_solve_qp(P, r, G, h, A, b)
        return cvxopt_solve_qp(P, r, G, h)
