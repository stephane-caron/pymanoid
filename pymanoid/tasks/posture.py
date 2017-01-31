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


class PostureTask(Task):

    """
    Track a set of reference joint angles, a common choice to regularize the
    weighted IK problem.
    """

    task_type = 'posture'

    def __init__(self, robot, q_ref, gain=None, weight=None, exclude_dofs=None):
        """
        Create task.

        INPUT:

        - ``robot`` -- a Robot object
        - ``q_ref`` -- vector of reference joint angles
        - ``gain`` -- (optional) residual gain between 0 and 1
        - ``weight`` -- task weight used in IK cost function
        - ``exclude_dofs`` -- (optional) DOFs not used by task
        """
        assert len(q_ref) == robot.nb_dofs

        J_posture = eye(robot.nb_dofs)
        if exclude_dofs is None:
            exclude_dofs = []
        if robot.has_free_flyer:  # don't include free-flyer coordinates
            exclude_dofs.extend([
                robot.TRANS_X, robot.TRANS_Y, robot.TRANS_Z, robot.ROT_Y])
        for i in exclude_dofs:
            J_posture[i, i] = 0.

        def pos_residual():
            e = (q_ref - robot.q)
            for j in exclude_dofs:
                e[j] = 0.
            return e

        def jacobian():
            return J_posture

        Task.__init__(
            self, jacobian, pos_residual=pos_residual, gain=gain, weight=weight,
            exclude_dofs=exclude_dofs)
