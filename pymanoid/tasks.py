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

from numpy import array, dot, eye, ndarray, zeros

_oppose_quat = array([-1., -1., -1., -1., +1., +1., +1.])


class Task(object):

    """
    Create a new IK task.

    Parameters
    ----------
    jacobian : function
        Function returning the task Jacobian when called with no argument.
    residual : function
        Function returning the task residual when called with no argument.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.

    Note
    ----
    Residuals returned by the ``residual`` function must have the unit of a
    velocity. For instance, ``qd`` and ``(q1 - q2) / dt`` are valid residuals,
    but ``0.5 * q`` is not.

    Notes
    -----
    See <https://scaron.info/teaching/inverse-kinematics.html> for an
    introduction to the concepts used here.
    """

    def __init__(self, jacobian, residual, weight, gain=0.85,
                 exclude_dofs=None):
        self._exclude_dofs = exclude_dofs
        self._jacobian = jacobian
        self._residual = residual
        self.gain = gain
        self.weight = weight

    def cost(self, dt):
        """
        Compute cost term of the task.

        Parameters
        ----------
        dt : scalar
            Time step in [s].

        Returns
        -------
        cost : scalar
            Current cost value.
        """
        r = self.residual(dt)
        return self.weight * dot(r, r)

    def exclude_dofs(self, dofs):
        """
        Exclude some DOFs from being used by the task.
        """
        if self._exclude_dofs is None:
            self._exclude_dofs = []
        self._exclude_dofs.extend(dofs)

    def jacobian(self):
        """
        Compute the Jacobian matrix of the task.

        Returns
        -------
        J : array
            Jacobian matrix of the task.
        """
        J = self._jacobian()
        if self._exclude_dofs:
            for dof_id in self._exclude_dofs:
                J[:, dof_id] *= 0.
        return J

    def residual(self, dt):
        return self.gain * self._residual(dt)


class COMTask(Task):

    """
    COM tracking task.

    Parameters
    ----------
    robot : Humanoid
        Targetted robot.
    target : list or array or Point
        Coordinates or the targetted COM position.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, target, weight, gain=0.85, exclude_dofs=None):
        self.update_target(target)
        residual = self._residual

        def jacobian():
            return self.robot.compute_com_jacobian()

        self.robot = robot
        self.name = 'com'
        super(COMTask, self).__init__(
            jacobian, residual, weight, gain, exclude_dofs)

    def update_target(self, target):
        if type(target) in [list, ndarray]:
            def residual(dt):
                return (target - self.robot.com) / dt
        elif hasattr(target, 'p'):
            def residual(dt):
                return (target.p - self.robot.com) / dt
        else:  # COM target should have a position field
            raise Exception("Target %s has no 'p' attribute" % type(target))
        self._residual = residual


class DOFTask(Task):

    """
    Track a reference DOF value.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    dof_id : string or integer
        DOF index or string of DOF identifier.
    dof_ref : scalar
        Reference DOF value.
    gain : scalar
        Proportional gain of the task.
    weight : scalar
        Task weight used in IK cost function.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, dof_id, dof_ref, weight, gain=0.85,
                 exclude_dofs=None):
        if type(dof_id) is str:
            dof_id = robot.__dict__[dof_id]
        J = zeros((1, robot.nb_dofs))
        J[0, dof_id] = 1.

        def residual(dt):
            return array([dof_ref - robot.q[dof_id]]) / dt

        def jacobian():
            return J

        self.dof_id = dof_id
        self.name = 'dof-%d' % dof_id
        super(DOFTask, self).__init__(
            jacobian, residual, weight, gain, exclude_dofs)


class LinkPosTask(Task):

    """
    Position task for a given link.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    link : Link
        One of the Link objects in the kinematic chain of the robot.
    target : list or array (shape=(3,)) or pymanoid.Body
        Coordinates of the link's target.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, link, target, weight, gain=0.85,
                 exclude_dofs=None):
        if type(link) is str:
            link = robot.__dict__[link]

        if hasattr(target, 'p'):
            def residual(dt):
                return (target.p - link.p) / dt
        elif type(target) in [list, ndarray]:
            def residual(dt):
                return (target - link.p) / dt
        else:  # this is an aesthetic comment
            raise Exception("Target %s has no 'p' attribute" % type(target))

        def jacobian():
            return robot.compute_link_pos_jacobian(link)

        self.link = link
        self.name = self.link.name
        super(LinkPosTask, self).__init__(
            jacobian, residual, weight, gain, exclude_dofs)


class LinkPoseTask(Task):

    """
    Pose task for a given link.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    link : Link
        One of the Link objects in the kinematic chain of the robot.
    target : list or array (shape=(7,)) or pymanoid.Body
        Pose coordinates of the link's target.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, link, target, weight, gain=0.85,
                 exclude_dofs=None):
        if type(link) is str:
            link = robot.__dict__[link]

        if hasattr(target, 'pose'):
            def residual(dt):
                pose_diff = target.pose - link.pose
                if dot(pose_diff[0:4], pose_diff[0:4]) > 1.:
                    pose_diff = _oppose_quat * target.pose - link.pose
                return pose_diff / dt
        elif type(target) in [list, ndarray]:
            def residual(dt):
                pose_diff = target - link.pose
                if dot(pose_diff[0:4], pose_diff[0:4]) > 1.:
                    pose_diff = _oppose_quat * target - link.pose
                return pose_diff / dt
        else:  # link frame target should be a pose
            raise Exception("Target %s has no 'pose' attribute" % type(target))

        def jacobian():
            return robot.compute_link_pose_jacobian(link)

        self.link = link
        self.name = self.link.name
        super(LinkPoseTask, self).__init__(
            jacobian, residual, weight, gain, exclude_dofs)


class MinAccelerationTask(Task):

    """
    Task to minimize joint accelerations.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.

    Note
    ----
    As the differential IK returns velocities, we approximate the task "minimize
    qdd" by "minimize (qd_next - qd)".
    """

    def __init__(self, robot, weight, gain=0.85, exclude_dofs=None):
        E = eye(robot.nb_dofs)

        def residual(dt):
            return robot.qd

        def jacobian():
            return E

        self.name = 'minaccel'
        super(MinAccelerationTask, self).__init__(
            self, jacobian, residual, weight, gain, exclude_dofs)


class MinCAMTask(Task):

    """
    Minimize the centroidal angular momentum (not its derivative).

    Parameters
    ----------
    robot : Humanoid
        Targetted robot.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, weight, gain=0.85, exclude_dofs=None):
        zero_cam = zeros((3,))

        def residual(dt):
            return zero_cam

        def jacobian():
            return robot.compute_cam_jacobian()

        self.name = 'mincam'
        super(MinCAMTask, self).__init__(
            jacobian, residual, weight, gain, exclude_dofs)


class MinVelocityTask(Task):

    """
    Task to minimize joint velocities

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, weight, gain=0.85, exclude_dofs=None):
        E = eye(robot.nb_dofs)
        qd_ref = zeros(robot.qd.shape)

        def residual(dt):
            return (qd_ref - robot.qd)

        def jacobian():
            return E

        self.name = 'minvel'
        super(MinVelocityTask, self).__init__(
            self, jacobian, residual, weight, gain, exclude_dofs)


class PendulumModeTask(Task):

    """
    Task to minimize the rate of change of the centroidal angular momentum.

    Parameters
    ----------
    robot : Humanoid
        Targetted robot.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.

    Notes
    -----
    The way this task is implemented may be surprising. Basically, keeping a
    constant CAM means d/dt(CAM) == 0, i.e.,

        d/dt (J_cam * qd) == 0
        J_cam * qdd + qd * H_cam * qd == 0

    Because the IK works at the velocity level, we approximate qdd by finite
    difference from the previous velocity (``qd`` argument to the residual
    function):

        J_cam * (qd_next - qd) / dt + qd * H_cam * qd == 0

    Finally, the task in qd_next (output velocity) is:

        J_cam * qd_next == J_cam * qd - dt * qd * H_cam * qd

    Hence, there are two occurrences of J_cam: one in the task residual, and the
    second in the task jacobian.
    """

    def __init__(self, robot, weight, gain=0.85, exclude_dofs=None):
        def residual(dt):
            qd = robot.qd
            J_cam = robot.compute_cam_jacobian()
            H_cam = robot.compute_cam_hessian()  # computation intensive :(
            return dot(J_cam, qd) - dt * dot(qd, dot(H_cam, qd))

        def jacobian():
            return robot.compute_cam_jacobian()

        self.name = 'pendulum'
        super(PendulumModeTask, self).__init__(
            jacobian, residual, weight, gain, exclude_dofs)


class PostureTask(Task):

    """
    Track a set of reference joint angles, a common choice to regularize the
    weighted IK problem.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    q_ref : array
        Vector of reference joint angles.
    weight : scalar
        Task weight used in IK cost function.
    gain : scalar
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, q_ref, weight, gain=0.85, exclude_dofs=None):
        assert len(q_ref) == robot.nb_dofs

        J_posture = eye(robot.nb_dofs)
        if exclude_dofs is None:
            exclude_dofs = []
        if robot.has_free_flyer:  # don't include free-flyer coordinates
            exclude_dofs.extend([
                robot.TRANS_X, robot.TRANS_Y, robot.TRANS_Z, robot.ROT_Y])
        for i in exclude_dofs:
            J_posture[i, i] = 0.

        def residual(dt):
            e = (q_ref - robot.q)
            for j in exclude_dofs:
                e[j] = 0.
            return e / dt

        def jacobian():
            return J_posture

        self.name = 'posture'
        super(PostureTask, self).__init__(
            jacobian, residual, weight, gain, exclude_dofs)


class ContactTask(LinkPoseTask):

    pass
