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


class ROSWrapper(object):

    """
    Update a robot's configuration from ROS topics
    """

    def __init__(self, robot, ignore_dofs=None):
        import rospy  # not global, will be initialized by child script
        self.dof_mapping = type(robot).__dict__
        self.flyer_tf = None
        self.ignore_dofs = set() if ignore_dofs is None else ignore_dofs
        self.map_tf = None
        self.robot = robot
        self.tf_listener = None
        self.zero_time = rospy.Time(0)

    def joint_state_callback(self, msg):
        """Update DOF values from a sensor_msgs/JointState."""
        q = self.robot.q
        for (i, joint_name) in enumerate(msg.name):
            dof = self.dof_mapping[joint_name]
            if dof not in self.ignore_dofs:
                q[dof] = msg.position[i]
        self.robot.set_dof_values(q)
        if self.tf_listener is not None:
            self.update_free_flyer()

    def set_free_flyer_tf(self, map_tf, flyer_tf):
        """Set ROS tf used to update the robot's free-flyer coordinates."""
        import tf  # not global, will be initialized by child script
        self.flyer_tf = flyer_tf
        self.map_tf = map_tf
        self.tf_listener = tf.TransformListener()

    def update_free_flyer(self):
        """Update free-flyer coordinates."""
        try:
            pos, rq = self.tf_listener.lookupTransform(
                self.map_tf, self.flyer_tf, self.zero_time)
            quat = [rq[3], rq[0], rq[1], rq[2]]  # ROS quat is (x, y, z, w)
            self.robot.set_free_flyer(pos, quat=quat)
        except:
            pass
