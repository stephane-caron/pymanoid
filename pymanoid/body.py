#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
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


import openravepy
import uuid

from numpy import array
from rotation import rotation_matrix_from_rpy, rpy_from_quat


class Body(object):

    """
    Wrapper for a RAVE object with the GetTransform() and GetTransformPose()
    methods, e.g. a Link or a Manipulator.
    """

    def __init__(self, rave_object, pos=None, rpy=None, color=None, name=None,
                 pose=None, visible=True, transparency=None):
        """
        Create body from an OpenRAVE KinBody.

        rave_object -- object to wrap
        pos -- initial position in inertial frame
        rpy -- initial orientation in inertial frame
        color -- color applied to all links (if any) in the object
        name -- object's name (optional)
        pose -- initial pose (supersedes pos and rpy)
        visible -- initial visibility
        transparency -- initial transparency (0 is opaque and 1 is transparent)
        """
        self.rave = rave_object
        if color is not None:
            self.set_color(color)
        if name is not None:
            self.rave.SetName(name)
        if pos is not None:
            self.set_pos(pos)
        if rpy is not None:
            self.set_rpy(rpy)
        if pose is not None:
            self.set_pose(pose)
        if not visible:
            self.set_visible(False)
        if transparency is not None:
            self.set_transparency(transparency)
        self.is_visible = visible

    def __str__(self):
        return "pymanoid.Body('%s')" % self.name

    def set_color(self, color):
        """
        Set the color of all bodies in the OpenRAVE KinBody object.

        color -- color code in Matplotlib convention
                 c.f. http://matplotlib.org/api/colors_api.html
        """
        if color == 'w':
            acolor = array([1., 1., 1.])
            dcolor = array([1., 1., 1.])
        else:  # add other colors above black
            acolor = array([.2, .2, .2])
            dcolor = array([.2, .2, .2])
            rgb, cmy = ['r', 'g', 'b'], ['c', 'm', 'y']
            if color in rgb:
                cdim = rgb.index(color)
                acolor[cdim] += .2
                dcolor[cdim] += .4
            elif color in cmy:
                cdim = cmy.index(color)
                acolor[(cdim + 1) % 3] += .2
                acolor[(cdim + 2) % 3] += .2
                dcolor[(cdim + 1) % 3] += .4
                dcolor[(cdim + 2) % 3] += .4
        for link in self.rave.GetLinks():
            for g in link.GetGeometries():
                g.SetAmbientColor(acolor)
                g.SetDiffuseColor(dcolor)

    def set_transparency(self, transparency):
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    def set_visible(self, visible):
        self.is_visible = visible
        self.rave.SetVisible(visible)

    @property
    def index(self):
        """Notably used to compute jacobians and hessians."""
        return self.rave.GetIndex()

    @property
    def name(self):
        """Get name from OpenRAVE object."""
        return self.rave.GetName()

    @property
    def T(self):
        """Transformation matrix."""
        return self.rave.GetTransform()

    @property
    def pose(self):
        """Pose (in OpenRAVE convention)."""
        pose = self.rave.GetTransformPose()
        if pose[0] < 0:  # convention: cos(alpha) > 0
            # this convention enforces Slerp shortest path
            pose[:4] *= -1
        return pose

    #
    # All other properties are derived from self.pose and self.T
    #

    @property
    def R(self):
        """Rotation matrix"""
        return self.T[0:3, 0:3]

    @property
    def p(self):
        """Position in world frame"""
        return self.T[0:3, 3]

    @property
    def pos(self):
        """Position in world frame"""
        return self.T[0:3, 3]

    @property
    def x(self):
        return self.p[0]

    @property
    def y(self):
        return self.p[1]

    @property
    def z(self):
        return self.p[2]

    @property
    def n(self):
        """Normal vector"""
        return self.T[0:3, 2]

    @property
    def quat(self):
        return self.pose[:4]

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

    #
    # Setters
    #

    def set_transform(self, T):
        self.rave.SetTransform(T)

    def set_pos(self, pos):
        T = self.T.copy()
        T[:3, 3] = pos
        self.set_transform(T)

    def set_x(self, x):
        T = self.T.copy()
        T[0, 3] = x
        self.set_transform(T)

    def set_y(self, y):
        T = self.T.copy()
        T[1, 3] = y
        self.set_transform(T)

    def set_z(self, z):
        T = self.T.copy()
        T[2, 3] = z
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

    #
    # Others
    #

    def remove(self):
        """Remove body from OpenRAVE environment."""
        env = self.rave.GetEnv()
        with env:
            env.Remove(self.rave)

    def __del__(self):
        """Add body removal to garbage collection step (effective)."""
        self.remove()


class Box(Body):

    def __init__(self, env, X, Y, Z, pos=None, rpy=None, color='r', name=None,
                 pose=None, visible=True, transparency=None):
        """
        Create a new box.

        env -- OpenRAVE environment
        X -- box half-length
        Y -- box half-width
        Z -- box half-height
        pos -- initial position in inertial frame
        rpy -- initial orientation in inertial frame
        color -- color letter in ['r', 'g', 'b']
        name -- object's name (optional)
        pose -- initial pose (supersedes pos and rpy)
        visible -- initial box visibility
        transparency -- initial transparency (0 is opaque and 1 is transparent)
        """
        if not name:
            name = "Box-%s" % str(uuid.uuid1())[0:3]
        self.X = X
        self.Y = Y
        self.Z = Z
        aabb = [0, 0, 0, X, Y, Z]
        box = openravepy.RaveCreateKinBody(env, '')
        box.InitFromBoxes(array([array(aabb)]), True)
        super(Box, self).__init__(
            box, pos=pos, rpy=rpy, color=color, name=name, pose=pose,
            visible=visible, transparency=transparency)
        env.Add(box, True)


class Cube(Box):

    def __init__(self, env, halflen, pos=None, rpy=None, color='r', name=None,
                 pose=None, visible=True, transparency=None):
        """
        Create a new box.

        env -- OpenRAVE environment
        halflen -- box half-length
        pos -- initial position in inertial frame
        rpy -- initial orientation in inertial frame
        color -- color letter in ['r', 'g', 'b']
        name -- object's name (optional)
        pose -- initial pose (supersedes pos and rpy)
        visible -- initial box visibility
        transparency -- initial transparency (0 is opaque and 1 is transparent)
        """
        super(Cube, self).__init__(
            env, halflen, halflen, halflen, pos=pos, rpy=rpy, color=color,
            name=name, pose=pose, visible=visible, transparency=transparency)


class Link(Body):

    def __init__(self, rave_link, color=None, pos=None, rpy=None,
                 pose=None, visible=True):
        super(Link, self).__init__(
            rave_link, color=color, pos=pos, rpy=rpy, pose=pose,
            visible=visible)


class Manipulator(Link):

    def __init__(self, rave_manipulator, color=None, pos=None,
                 rpy=None, pose=None, visible=True):
        super(Manipulator, self).__init__(
            rave_manipulator, color=color, pos=pos, rpy=rpy, pose=pose,
            visible=visible)
        self.end_effector = rave_manipulator.GetEndEffector()

    def set_transparency(self, transparency):
        print "Warning:", \
            "manipulators have no link (called from %s) " % self.name

    def set_visible(self, visible):
        print "Warning:", \
            "manipulators have no visibility (called from %s)" % self.name

    @property
    def index(self):
        """Notably used to compute jacobians and hessians."""
        return self.end_effector.GetIndex()
