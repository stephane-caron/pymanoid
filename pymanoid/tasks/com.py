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

from numpy import array, ndarray

from task import Task


class COMTask(Task):

    task_type = 'com'

    def __init__(self, robot, target, gain=None, weight=None,
                 exclude_dofs=None):
        """
        Add a COM tracking task.

        INPUT:

        - ``robot`` -- a CentroidalRobot object
        - ``target`` -- coordinates or any Body object with a 'pos' attribute
        """
        self.robot = robot
        jacobian = self.robot.compute_com_jacobian
        pos_residual = self.compute_pos_residual(target)
        Task.__init__(
            self, jacobian, pos_residual=pos_residual, gain=gain, weight=weight,
            exclude_dofs=exclude_dofs)

    def compute_pos_residual(self, target):
        if type(target) is list:
            target = array(target)
        if type(target) is ndarray:
            def pos_residual():
                return target - self.robot.com
        elif hasattr(target, 'p'):
            def pos_residual():
                return target.p - self.robot.com
        else:  # COM target should be a position
            raise Exception("Target %s has no 'p' attribute" % type(target))
        return pos_residual

    def update_target(self, target):
        self.pos_residual = self.compute_pos_residual(target)
