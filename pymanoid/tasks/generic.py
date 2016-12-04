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

    # NB: a proportional gain of 1. can cause oscillations
    # when some tasks are unfeasible
    DEFAULT_GAIN = 0.85

    task_type = 'generic'

    def __init__(self, jacobian, pos_residual=None, vel_residual=None,
                 gain=None, weight=None, exclude_dofs=None):
        """
        Create a new IK task.

        INPUT:

        - ``pos_residual`` -- position residual
        - ``vel_residual`` -- velocity residual (excludes position one)
        - ``gain`` -- (optional) residual gain between 0 and 1
        - ``weight`` -- task weight used in IK cost function
        - ``exclude_dofs`` -- (optional) DOFs not used by task

        .. NOTE::

            See <https://scaron.info/teaching/inverse-kinematics.html> for
            details on what the "position" and "velocity" residuals are.
        """
        assert pos_residual or vel_residual
        self.__jacobian = jacobian
        self.__exclude_dofs = exclude_dofs
        self.gain = gain if gain is not None else self.DEFAULT_GAIN
        self.pos_residual = pos_residual
        self.vel_residual = vel_residual
        self.weight = weight

    @property
    def name(self):
        return self.task_type

    def check(self):
        """Check that gain and weights are consistent."""
        if self.weight is None:
            raise Exception(
                "No weight supplied for task '%s' of type '%s'" % (
                    self.name, type(self).task_type))
        if not (0. <= self.gain <= 1.):
            raise Exception("Gain for task '%s' should be in (0, 1)" % (
                self.name, self.gain))

    def cost(self, dt):
        """
        Compute cost term of the task.

        INPUT:

        ``dt`` -- IK time step
        """
        def sq(r):
            return dot(r, r)
        return self.weight * sq(self.residual(dt))

    def exclude_dofs(self, dofs):
        """Exclude some DOFs from being used by the task."""
        if self.__exclude_dofs is None:
            self.__exclude_dofs = []
        self.__exclude_dofs.extend(dofs)

    def jacobian(self):
        """
        Compute task Jacobian matrix.

        INPUT:

        ``dt`` -- IK time step
        """
        J = self.__jacobian()
        if self.__exclude_dofs:
            for dof_id in self.__exclude_dofs:
                J[:, dof_id] *= 0.
        return J

    def residual(self, dt):
        """
        Compute task residual vector.

        INPUT:

        ``dt`` -- IK time step
        """
        if self.vel_residual is not None:
            return self.gain * self.vel_residual(dt)
        else:  # use position residual
            return self.gain * self.pos_residual() / dt
