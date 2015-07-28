#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of openravepypy.
#
# openravepypy is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# openravepypy is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openravepypy. If not, see <http://www.gnu.org/licenses/>.


import openravepy
import uuid

from rotation import rotation_matrix_from_rpy, rpy_from_quat
from numpy import array, dot


class Body(object):

    def __init__(self, env, name, rave_body, color=None, pos=None, rpy=None,
                 pose=None):
        self.rave = rave_body
        self.rave.SetName(name)
        env.Add(self.rave, True)
        if color is not None:
            self.set_color(color)
        if pos is not None:
            self.set_pos(pos)
        if rpy is not None:
            self.set_rpy(rpy)
        if pose is not None:
            self.set_pose(pose)

    def set_color(self, color):
        acolor = array([.2, .2, .2])
        dcolor = array([.2, .2, .2])
        colors = ['r', 'g', 'b']
        if color in colors:
            cdim = colors.index(color)
            acolor[cdim] += .2
            dcolor[cdim] += .4
        for link in self.rave.GetLinks():
            for g in link.GetGeometries():
                g.SetAmbientColor(acolor)
                g.SetDiffuseColor(dcolor)

    def set_transparency(self, transparency):
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    @property
    def pose(self):
        pose = self.rave.GetTransformPose()
        if pose[0] < 0:  # convention: cos(alpha) > 0
            # this convention enforces Slerp shortest path
            pose[:4] *= -1
        return pose

    @property
    def quat(self):
        return self.pose[:4]

    @property
    def T(self):
        """Transform matrix"""
        return self.rave.GetTransform()

    @property
    def R(self):
        """Rotation matrix"""
        return self.T[0:3, 0:3]

    @property
    def p(self):
        """Position"""
        return self.T[0:3, 3]

    @property
    def n(self):
        """Normal vector"""
        return self.R[2, 0:3]

    @property
    def rpy(self):
        """Roll-pitch-yaw angles"""
        return rpy_from_quat(self.quat)

    @property
    def roll(self):
        return self.rpy[0]

    @property
    def pitch(self):
        return self.rpy[1]

    @property
    def yaw(self):
        return self.rpy[2]

    def set_transform(self, T):
        self.rave.SetTransform(T)

    def set_pos(self, pos):
        T = self.T.copy()
        T[:3, 3] = pos
        self.set_transform(T)

    def set_rpy(self, rpy):
        T = self.T.copy()
        T[0:3, 0:3] = rotation_matrix_from_rpy(*rpy)
        self.set_transform(T)

    def set_roll(self, roll):
        return self.set_rpy([roll, self.pitch, self.yaw])

    def set_pitch(self, pitch):
        return self.set_rpy([self.roll, pitch, self.yaw])

    def set_yaw(self, yaw):
        return self.set_rpy([self.roll, self.pitch, yaw])

    def set_pose(self, pose):
        T = openravepy.matrixFromPose(pose)
        self.set_transform(T)


class Box(Body):

    def __init__(self, env, box_dim=None, X=None, Y=None, Z=None, color='r',
                 name=None, pos=None, rpy=None, pose=None):
        if not name:
            name = "Box-%s" % str(uuid.uuid1())[0:3]
        if box_dim is not None:
            X = box_dim
            Y = box_dim
            Z = box_dim
        self.X = X
        self.Y = Y
        self.Z = Z
        aabb = [0, 0, 0, X, Y, Z]
        rave_body = openravepy.RaveCreateKinBody(env, '')
        rave_body.InitFromBoxes(array([array(aabb)]), True)
        super(Box, self).__init__(env, name, rave_body, color, pos=pos, rpy=rpy,
                                  pose=pose)
