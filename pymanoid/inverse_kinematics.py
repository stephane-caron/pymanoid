#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
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
from threading import Lock
from utils import cvxopt_solve_qp
from uuid import uuid1


class DiffIKSolver(object):

    class Task(object):

        def __init__(self, error, jacobian, gain, weight=None):
            self.error = error
            self.gain = gain
            self.jacobian = jacobian
            self.weight = weight

        def jacobian(self, q):
            return self.task.jacobian(q)

        def value(self, q, qd):
            e = self.error(q, qd)
            return self.weight * dot(e, e)

        def velocity(self, q, qd):
            return self.gain * self.error(q, qd)

    def __init__(self, robot, qd_lim, K_doflim=None):
        n = len(robot.q)
        self.I = eye(n)
        self.K_doflim = K_doflim
        self.constraints = []
        self.lock = Lock()
        self.objectives = {}
        self.q_max = robot.q_max
        self.q_min = robot.q_min
        self.qd_max = +qd_lim * ones(n)
        self.qd_min = -qd_lim * ones(n)

    def add_constraint(self, error, jacobian, gain):
        self.lock.acquire()
        self.constraints.append(self.Task(error, jacobian, gain))
        self.lock.release()

    def add_objective(self, error, jacobian, gain, weight, name=None):
        if not name:
            name = "Objective-%s" % str(uuid1())[0:3]
        self.lock.acquire()
        self.objectives[name] = self.Task(error, jacobian, gain, weight)
        self.lock.release()

    def remove_objective(self, name):
        del self.objectives[name]

    def compute_objective(self, q, qd):
        return sum(obj.value(q, qd) for obj in self.objectives.itervalues())

    def compute_velocity(self, q, qd):
        self.lock.acquire()
        P = zeros(self.I.shape)
        r = zeros(self.q_max.shape)
        for obj in self.objectives.itervalues():
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
            self.lock.release()
            return cvxopt_solve_qp(P, r, G, h, A, b)
        self.lock.release()
        return cvxopt_solve_qp(P, r, G, h)
