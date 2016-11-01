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

from numpy import array, eye, zeros

from generic import Task


class DOFTask(Task):

    """Track a reference DOF value"""

    task_type = 'dof'

    def __init__(self, robot, dof_id, dof_ref, **kwargs):
        """
        Create task.

        INPUT:

        - ``robot`` -- a Robot object
        """
        J = zeros((1, robot.nb_dofs))
        J[0, dof_id] = 1.

        def pos_residual():
            return array([dof_ref - robot.q[dof_id]])

        def jacobian():
            return J

        self.dof_id = dof_id
        super(DOFTask, self).__init__(
            jacobian, pos_residual=pos_residual, **kwargs)

    @property
    def name(self):
        return 'dof-%d' % self.dof_id


class MinAccelerationTask(Task):

    """Minimize joint accelerations"""

    task_type = 'minaccel'

    def __init__(self, robot, **kwargs):
        """
        Create task.

        INPUT:

        - ``robot`` -- a Robot object

        .. NOTE::

            As the differential IK returns velocities, we approximate the task
            "minimize qdd" by "minimize (qd_next - qd)".
        """
        E = eye(robot.nb_dofs)

        def vel_residual(dt):
            return robot.qd

        def jacobian():
            return E

        Task.__init__(self, jacobian, vel_residual=vel_residual, **kwargs)


class MinVelocityTask(Task):

    """Minimize joint velocities"""

    task_type = 'minvel'

    def __init__(self, robot, **kwargs):
        """
        Create task.

        INPUT:

        - ``robot`` -- a Robot object
        """
        E = eye(robot.nb_dofs)

        def vel_residual(dt):
            return -robot.qd

        def jacobian():
            return E

        Task.__init__(self, jacobian, vel_residual=vel_residual, **kwargs)


class PostureTask(Task):

    """
    Track a set of reference joint angles, a common choice to regularize the
    weighted IK problem.
    """

    task_type = 'posture'

    def __init__(self, robot, q_ref, exclude_dofs=None, **kwargs):
        """
        Create task.

        INPUT:

        - ``robot`` -- a Robot object
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

        Task.__init__(self, jacobian, pos_residual=pos_residual, **kwargs)
