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


from numpy import dot, eye, hstack, maximum, minimum, ones, vstack, zeros
from qp import solve_qp
from threading import Lock
from warnings import warn


class DiffIKSolver(object):

    def __init__(self, dt, q_max, q_min, qd_lim, K_doflim, gains=None,
                 weights=None):
        n = len(q_max)
        self.I = eye(n)
        self.K_doflim = K_doflim
        self.dt = dt
        self.errors = {}
        self.gains = {}
        self.jacobians = {}
        self.q_max = q_max
        self.q_min = q_min
        self.qd_max = +qd_lim * ones(n)
        self.qd_min = -qd_lim * ones(n)
        self.tasks = {}
        self.tasks_lock = Lock()
        self.weights = {}
        if gains is not None:
            self.gains.update(gains)
        if weights is not None:
            self.weights.update(weights)

    def add_task(self, name, error, jacobian, gain=None, weight=None,
                 task_type=None):
        assert name not in self.tasks, \
            "Task '%s' already present in IK" % name
        with self.tasks_lock:
            self.tasks[name] = True
            self.errors[name] = error
            self.jacobians[name] = jacobian
            if gain is not None:
                self.gains[name] = gain
            elif name in self.gains:
                pass  # gain is already defined
            elif task_type in self.gains:
                self.gains[name] = self.gains[task_type]
            else:
                msg = "No gain provided for task '%s'" % name
                if task_type is not None:
                    msg += " (task_type='%s')" % task_type
                raise Exception(msg)
            if weight is not None:
                self.weights[name] = weight
            elif name in self.weights:
                pass   # weight is already defined
            elif task_type in self.weights:
                self.weights[name] = self.weights[task_type]
            else:
                msg = "No weight provided for task '%s'" % name
                if task_type is not None:
                    msg += " (task_type='%s')" % task_type
                raise Exception(msg)

    def remove_task(self, name):
        with self.tasks_lock:
            if name not in self.tasks:
                warn("no task '%s' to remove" % name)
                return
            del self.tasks[name]

    def update_gain(self, name, gain):
        self.gains[name] = gain

    def update_weight(self, name, weight):
        self.weights[name] = weight

    def compute_cost(self, q, qd):
        def sq(e):
            return dot(e, e)

        def cost(task):
            return self.weights[task] * sq(self.errors[task](q, qd))

        return sum(cost(task) for task in self.tasks)

    def compute_velocity(self, q, qd):
        P = zeros(self.I.shape)
        r = zeros(self.q_max.shape)
        with self.tasks_lock:
            for task in self.tasks:
                J = self.jacobians[task](q)
                e = self.gains[task] * self.errors[task](q, qd)
                P += self.weights[task] * dot(J.T, J)
                r += self.weights[task] * dot(-e.T, J)
        qd_max = minimum(self.qd_max, self.K_doflim * (self.q_max - q))
        qd_min = maximum(self.qd_min, self.K_doflim * (self.q_min - q))
        G = vstack([+self.I, -self.I])
        h = hstack([qd_max, -qd_min])
        return solve_qp(P, r, G, h)
