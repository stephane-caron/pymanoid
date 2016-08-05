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

from numpy import dot


class Task(object):

    task_type = 'generic'

    def __init__(self, jacobian, pos_residual=None, vel_residual=None,
                 gain=None, weight=None, exclude_dofs=None):
        """
        Create a new IK task.

        INPUT:

        - ``pos_residual``:
        """
        assert pos_residual or vel_residual
        self.__jacobian = jacobian
        self.__exclude_dofs = exclude_dofs
        self.gain = gain
        self.pos_residual = pos_residual
        self.vel_residual = vel_residual
        self.weight = weight

    @property
    def name(self):
        return self.task_type

    def cost(self, dt):
        def sq(r):
            return dot(r, r)

        return self.weight * sq(self.residual(dt))

    def exclude_dofs(self, dofs):
        if self.__exclude_dofs is None:
            self.__exclude_dofs = []
        self.__exclude_dofs.extend(dofs)

    def jacobian(self):
        J = self.__jacobian()
        if self.__exclude_dofs:
            for dof_id in self.__exclude_dofs:
                J[:, dof_id] *= 0.  # we are working on the full jacobian
        return J

    def residual(self, dt):
        vel_residual = \
            self.vel_residual(dt) if self.vel_residual is not None else \
            self.pos_residual() / dt
        if self.gain is None:
            return vel_residual
        return self.gain * vel_residual
