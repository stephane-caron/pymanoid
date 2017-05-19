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

from numpy import array, dot, eye, zeros

from misc import PointWrap, PoseWrap


DEFAULT_GAINS = {
    'COM': 0.85,
    'CONTACT': 0.85,
    'DOF': 0.85,
    'EFFECTOR': 0.85,
    'POSTURE': 0.85,
    'REGULARIZATION': 0.85,
}
DEFAULT_WEIGHTS = {
    'CONTACT': 1.,
    'COM': 1e-2,
    'DOF': 1e-5,
    'EFFECTOR': 1e-3,
    'POSTURE': 1e-6,
    'REGULARIZATION': 1e-6,
}
_OPPOSE_QUAT = array([-1., -1., -1., -1., +1., +1., +1.])


class Task(object):

    """
    Generic IK task.

    Parameters
    ----------
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.

    Note
    ----
    See <https://scaron.info/teaching/inverse-kinematics.html> for an
    introduction to the concepts used here.
    """

    def __init__(self, weight=None, gain=None, exclude_dofs=None):
        self._exclude_dofs = [] if exclude_dofs is None else exclude_dofs
        self.gain = gain
        self.weight = weight

    def cost(self, dt):
        """
        Compute the weighted norm of the task residual.

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
        Exclude additional DOFs from being used by the task.
        """
        self._exclude_dofs.extend(dofs)

    def _jacobian(self):
        raise NotImplementedError("Task Jacobian not implemented")

    def jacobian(self):
        """
        Compute the Jacobian matrix of the task.

        Returns
        -------
        J : array
            Jacobian matrix of the task.
        """
        J = self._jacobian()
        for dof_id in self._exclude_dofs:
            J[:, dof_id] *= 0.
        return J

    def _residual(self, dt):
        raise NotImplementedError("Task residual not implemented")

    def residual(self, dt):
        """
        Compute the residual of the task.

        Returns
        -------
        r : array
            Residual vector of the task.

        Notes
        -----
        Residuals returned by the ``residual`` function must have the unit of a
        velocity. For instance, :math:`\\dot{q}` and :math:`(q_1 - q_2) / dt`
        are valid residuals, but :math:`\\frac12 q` is not.
        """
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
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, target, weight=None, gain=None,
                 exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['COM']
        if weight is None:
            weight = DEFAULT_WEIGHTS['COM']
        super(COMTask, self).__init__(weight, gain, exclude_dofs)
        self.name = 'COM'
        self.robot = robot
        self.update_target(target)

    def _jacobian(self):
        return self.robot.compute_com_jacobian()

    def _residual(self, dt):
        return (self.target.p - self.robot.com) / dt

    def update_target(self, target):
        """
        Update the task residual with a new target.

        Parameters
        ----------
        target : Point or array or list
            New COM position target.
        """
        self.target = target if hasattr(target, 'p') else PointWrap(target)


class COMAccelTask(Task):

    """
    COM acceleration task.

    Parameters
    ----------
    robot : Humanoid
        Targetted robot.
    target : list or array or Point
        Coordinates or the targetted COM position.
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.

    Notes
    -----
    Expanding :math:`\\ddot{x}_G = u` in terms of COM Jacobian and Hessian, the
    equation of the task is:

    .. math::

        J_\\mathrm{COM} \\dot{q}_\\mathrm{next}
        = \\frac12 u + J_\\mathrm{COM} \\dot{q} - \\frac12 \\delta t \\dot{q}^T
        H_\\mathrm{COM} \\dot{q}

    See the documentation of the PendulumModeTask for a detailed derivation.
    """

    def __init__(self, robot, weight=None, gain=None, exclude_dofs=None):
        super(COMAccelTask, self).__init__(weight, gain, exclude_dofs)
        self._comdd = zeros(3)
        self.name = 'COM_ACCEL'
        self.robot = robot

    def _jacobian(self):
        return self.robot.compute_com_jacobian()

    def _residual(self, dt):
        next_comd = self.robot.comd + dt * self._comdd
        return next_comd

    def update_command(self, comdd):
        self._comdd = comdd


class DOFTask(Task):

    """
    Track a reference DOF value.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    index : string or integer
        DOF index or string of DOF identifier.
    target : scalar
        Target DOF value.
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, index, target, weight=None, gain=None,
                 exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['DOF']
        if weight is None:
            weight = DEFAULT_WEIGHTS['DOF']
        super(DOFTask, self).__init__(weight, gain, exclude_dofs)
        if type(index) is str:
            index = robot.__getattribute__(index)
        J = zeros((1, robot.nb_dofs))
        J[0, index] = 1.
        self.__J = J
        self.index = index
        self.name = robot.get_dof_name_from_index(index)
        self.robot = robot
        self.target = target

    def _jacobian(self):
        return self.__J

    def _residual(self, dt):
        return array([self.target - self.robot.q[self.index]]) / dt

    def update_target(self, target):
        self.target = target


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
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, link, target, weight=None, gain=None,
                 exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['EFFECTOR']
        if weight is None:
            weight = DEFAULT_WEIGHTS['EFFECTOR']
        super(LinkPosTask, self).__init__(weight, gain, exclude_dofs)
        if type(link) is str:
            link = robot.__getattribute__(link)
        self.link = link
        self.name = self.link.name
        self.robot = robot
        self.update_target(target)

    def _jacobian(self):
        return self.robot.compute_link_pos_jacobian(self.link)

    def _residual(self, dt):
        return (self.target.p - self.link.p) / dt

    def update_target(self, target):
        """
        Update the task residual with a new target.

        Parameters
        ----------
        target : Point or array or list
            New link position target.
        """
        self.target = target if hasattr(target, 'p') else PointWrap(target)


class LinkPoseTask(Task):

    """
    Pose task for a given link.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    link : Link or string
        Robot Link, or name of the link field in the ``robot`` argument.
    target : list or array (shape=(7,)) or pymanoid.Body
        Pose coordinates of the link's target.
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, link, target, weight=None, gain=None,
                 exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['EFFECTOR']
        if weight is None:
            weight = DEFAULT_WEIGHTS['EFFECTOR']
        super(LinkPoseTask, self).__init__(weight, gain, exclude_dofs)
        if type(link) is str:
            link = robot.__getattribute__(link)
        self.link = link
        self.name = self.link.name
        self.robot = robot
        self.update_target(target)

    def _jacobian(self):
        return self.robot.compute_link_pose_jacobian(self.link)

    def _residual(self, dt):
        pose_diff = self.target.pose - self.link.pose
        if dot(pose_diff[0:4], pose_diff[0:4]) > 1.:
            pose_diff = _OPPOSE_QUAT * self.target.pose - self.link.pose
        return pose_diff / dt

    def update_target(self, target):
        """
        Update the task residual with a new target.

        Parameters
        ----------
        target : Point or array or list
            New link position target.
        """
        self.target = target if hasattr(target, 'pose') else PoseWrap(target)


class MinAccelTask(Task):

    """
    Task to minimize joint accelerations.

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.

    Note
    ----
    As the differential IK returns velocities, we approximate the minimization
    over :math:`\\ddot{q}` by that over :math:`(\\dot{q}_\\mathrm{next} -
    \\dot{q})`. See the documentation of the PendulumModeTask for details on the
    discrete approximation of :math:`\\ddot{q}`.
    """

    def __init__(self, robot, weight=None, gain=None, exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['REGULARIZATION']
        if weight is None:
            weight = DEFAULT_WEIGHTS['REGULARIZATION']
        super(MinAccelTask, self).__init__(weight, gain, exclude_dofs)
        self.__J = eye(robot.nb_dofs)
        self.name = 'MIN_ACCEL'
        self.robot = robot

    def _jacobian(self):
        return self.__J

    def _residual(self, dt):
        return self.robot.qd


class MinCAMTask(Task):

    """
    Minimize the centroidal angular momentum (not its derivative).

    Parameters
    ----------
    robot : Humanoid
        Targetted robot.
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, weight=None, gain=None, exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['REGULARIZATION']
        if weight is None:
            weight = DEFAULT_WEIGHTS['REGULARIZATION']
        super(MinCAMTask, self).__init__(weight, gain, exclude_dofs)
        self.__zero_cam = zeros((3,))
        self.name = 'MIN_CAM'
        self.robot = robot

    def _jacobian(self):
        return self.robot.compute_cam_jacobian()

    def _residual(self, dt):
        return self.__zero_cam


class MinVelTask(Task):

    """
    Task to minimize joint velocities

    Parameters
    ----------
    robot : Robot
        Targetted robot.
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, weight=None, gain=None, exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['REGULARIZATION']
        if weight is None:
            weight = DEFAULT_WEIGHTS['REGULARIZATION']
        super(MinVelTask, self).__init__(weight, gain, exclude_dofs)
        self.__J = eye(robot.nb_dofs)
        self.name = 'MIN_VEL'
        self.qd_ref = zeros(robot.qd.shape)
        self.robot = robot

    def _jacobian(self):
        return self.__J

    def _residual(self, dt):
        return self.qd_ref - self.robot.qd


class PendulumModeTask(Task):

    """
    Task to minimize the rate of change of the centroidal angular momentum.

    Parameters
    ----------
    robot : Humanoid
        Targetted robot.
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.

    Notes
    -----
    The way this task is implemented may be surprising. Basically, keeping a
    constant CAM :math:`L_G` means that :math:`\\dot{L}_G = 0`, that is,

    .. math::

        \\frac{\\mathrm{d} (J_\\mathrm{CAM} \\dot{q})}{\\mathrm{d} t}
        = 0 \\ \\Leftrightarrow\\
        J_\\mathrm{CAM} \\ddot{q} + \\dot{q}^T H_\\mathrm{CAM} \\dot{q}
        = 0

    Because the IK works at the velocity level, we approximate :math:`\\ddot{q}`
    by finite difference from the previous robot velocity. Assuming that the
    velocity :math:`\\dot{q}_\\mathrm{next}` output by the IK is applied
    immediately, joint angles become:

    .. math::

        q' = q + \\dot{q}_\\mathrm{next} \\delta t

    Meanwhile, the Taylor expansion of `q` is

    .. math::

        q' = q + \\dot{q} \\delta t + \\frac12 \\ddot{q} \delta t^2,

    so that applying :math:`\\dot{q}_\\mathrm{next}` is equivalent to having the
    following constant acceleration over :math:`\\delta t`:

    .. math::

        \\ddot{q} = 2 \\frac{\\dot{q}_\\mathrm{next} - \\dot{q}}{\\delta t}.

    Replacing in the Jacobian/Hessian expansion yields:

    .. math::

        2 J_\\mathrm{CAM} \\frac{\\dot{q}_\\mathrm{next} - \\dot{q}}{\\delta t}
        + + \\dot{q}^T H_\\mathrm{CAM} \\dot{q} = 0.

    Finally, the task in :math:`\\dot{q}_\\mathrm{next}` is:

    .. math::

        J_\\mathrm{CAM} \\dot{q}_\\mathrm{next}
        = J_\\mathrm{CAM} \\dot{q} - \\frac12 \\delta t \\dot{q}^T
        H_\\mathrm{CAM} \\dot{q}

    Note how there are two occurrences of :math:`J_\\mathrm{CAM}`: one in the
    task residual, and the second in the task Jacobian.
    """

    def __init__(self, robot, weight=None, gain=None, exclude_dofs=None):
        super(PendulumModeTask, self).__init__(weight, gain, exclude_dofs)
        self.name = 'PENDULUM'
        self.robot = robot

    def _jacobian(self):
        return self.robot.compute_cam_jacobian()

    def _residual(self, dt):
        qd = self.robot.qd
        J_cam = self.robot.compute_cam_jacobian()
        H_cam = self.robot.compute_cam_hessian()  # computation intensive :(
        return dot(J_cam, qd) - .5 * dt * dot(qd, dot(H_cam, qd))


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
    weight : scalar, optional
        Task weight used in IK cost function. If None, needs to be set later.
    gain : scalar, optional
        Proportional gain of the task.
    exclude_dofs : list of integers, optional
        DOF indices not used by task.
    """

    def __init__(self, robot, q_ref, weight=None, gain=None, exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['REGULARIZATION']
        if weight is None:
            weight = DEFAULT_WEIGHTS['REGULARIZATION']
        super(PostureTask, self).__init__(weight, gain, exclude_dofs)
        J = eye(robot.nb_dofs)
        if exclude_dofs is None:
            exclude_dofs = []
        if robot.has_free_flyer:  # don't include free-flyer coordinates
            exclude_dofs.extend([
                robot.TRANS_X, robot.TRANS_Y, robot.TRANS_Z, robot.ROT_Y])
        for i in exclude_dofs:
            J[i, i] = 0.
        self.__J = J
        self.name = 'POSTURE'
        self.q_ref = q_ref
        self.robot = robot

    def _jacobian(self):
        return self.__J

    def _residual(self, dt):
        e = self.q_ref - self.robot.q
        for i in self._exclude_dofs:
            e[i] = 0.
        return e / dt


class ContactTask(LinkPoseTask):

    """
    In essence, a contact task is a :class:`pymanoid.tasks.LinkPoseTask` with a
    much (much) larger weight. For plausible motions, no task should have a
    weight higher than (or even comparable to) that of contact tasks.
    """

    def __init__(self, robot, link, target, weight=None, gain=None,
                 exclude_dofs=None):
        if gain is None:
            gain = DEFAULT_GAINS['CONTACT']
        if weight is None:
            weight = DEFAULT_WEIGHTS['CONTACT']
        super(ContactTask, self).__init__(
            robot, link, target, weight, gain, exclude_dofs)
