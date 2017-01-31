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

from numpy import eye

from task import Task


class MinAccelerationTask(Task):

    """Minimize joint accelerations"""

    task_type = 'minaccel'

    def __init__(self, robot, gain=None, weight=None, exclude_dofs=None):
        """
        Create task.

        INPUT:

        - ``robot`` -- a Robot object
        - ``gain`` -- (optional) residual gain between 0 and 1
        - ``weight`` -- task weight used in IK cost function
        - ``exclude_dofs`` -- (optional) DOFs not used by task

        .. NOTE::

            As the differential IK returns velocities, we approximate the task
            "minimize qdd" by "minimize (qd_next - qd)".
        """
        E = eye(robot.nb_dofs)

        def vel_residual(dt):
            return robot.qd

        def jacobian():
            return E

        Task.__init__(
            self, jacobian, vel_residual=vel_residual, gain=gain, weight=weight,
            exclude_dofs=exclude_dofs)
