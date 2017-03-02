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


from geometry_msgs.msg import PointStamped
from geometry_msgs.msg import TransformStamped


def PointStamped_from_pos(p, seq=None):
    """
    Create a geometry_msgs/PointStamped from position coordinates.

    Parameters
    ----------
    pos : ndarray
        Coordinates in world frame.
    seq : integer
        Sequence number for transform's header.
    """
    ps = PointStamped()
    if seq is not None:
        ps.header.seq = seq
    ps.point.x = p[0]
    ps.point.y = p[1]
    ps.point.z = p[2]
    return ps


def TransformStamped_from_pose(pose, seq=None):
    """
    Create a geometry_msgs/TransformStamped from a given pose.

    Parameters
    ----------
    pose : array, shape=(7,)
        Pose in OpenRAVE format.
    seq : integer, optional
        Sequence number for the transform header.
    """
    ts = TransformStamped()
    if seq is not None:
        ts.header.seq = seq
    ts.transform.rotation.w = pose[0]
    ts.transform.rotation.x = pose[1]
    ts.transform.rotation.y = pose[2]
    ts.transform.rotation.z = pose[3]
    ts.transform.translation.x = pose[4]
    ts.transform.translation.y = pose[5]
    ts.transform.translation.z = pose[6]
    return ts


class ROSWrapper(object):

    """
    Update a robot's configuration from ROS topics
    """

    def __init__(self, robot, ignore_dofs=None):
        """
        Update robot configuration from ROS topics and tfs.

        Parameters
        ----------
        robot : pymanoid.Robot
            Wrapped robot model.
        ignore_dofs : list of integers
            List of DOFs not updated by ROS.
        """
        import rospy  # not global, will be initialized by child script
        self.dof_mapping = type(robot).__dict__
        self.flyer_tf = None
        self.ignore_dofs = set() if ignore_dofs is None else ignore_dofs
        self.map_tf = None
        self.robot = robot
        self.tf_listener = None
        self.zero_time = rospy.Time(0)

    def joint_state_callback(self, msg):
        """
        Update DOF values from a sensor_msgs/JointState.

        Parameters
        ----------
        msg : sensor_msgs.JointState
            Callback message.
        """
        q = self.robot.q
        for (i, joint_name) in enumerate(msg.name):
            dof = self.dof_mapping[joint_name]
            if dof not in self.ignore_dofs:
                q[dof] = msg.position[i]
        self.robot.set_dof_values(q, clamp=True)
        if self.tf_listener is not None:
            self.update_free_flyer()

    def set_free_flyer_tf(self, map_tf, flyer_tf):
        """
        Set ROS tf used to update the robot's free-flyer coordinates.

        Parameters
        ----------
        map_tf : string
            Name of the world frame.
        flyer_tf : string
            Name of the free-flyer frame.
        """
        import tf  # not global, will be initialized by child script
        self.flyer_tf = flyer_tf
        self.map_tf = map_tf
        self.tf_listener = tf.TransformListener()

    def update_free_flyer(self):
        """
        Update free-flyer coordinates.
        """
        try:
            pos, rq = self.tf_listener.lookupTransform(
                self.map_tf, self.flyer_tf, self.zero_time)
            quat = [rq[3], rq[0], rq[1], rq[2]]  # ROS quat is (x, y, z, w)
            self.robot.set_ff_pos(pos)
            self.robot.set_ff_quat(quat)
        except Exception as e:
            print "update_free_flyer():", e
