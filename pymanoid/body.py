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

import openravepy

from numpy import array, dot, ndarray, zeros

from draw import matplotlib_to_rgb
from misc import norm
from rotations import crossmat, rotation_matrix_from_rpy, rpy_from_quat
from sim import get_openrave_env


class Body(object):

    """
    Base class for rigid bodies. Wraps OpenRAVE's KinBody type.

    Parameters
    ----------
    kinbody : openravepy.KinBody
        OpenRAVE body to wrap.
    pos : array, shape=(3,), optional
        Initial position in inertial frame.
    rpy : array, shape=(3,), optional
        Initial orientation in inertial frame.
    pose : array, shape=(7,), optional
        Initial pose. Supersedes ``pos`` and ``rpy`` if they are provided at
        the same time.
    color : char, optional
        Color code in `Matplotlib convention
        <http://matplotlib.org/api/colors_api.html>`_.
    visible : bool, optional
        Initial visibility.
    transparency : double, optional
        Transparency value from 0 (opaque) to 1 (invisible).
    name : string, optional
        Body name in OpenRAVE scope.
    """

    count = 0
    """
    Counter for anonymous bodies.
    """

    def __init__(self, kinbody, pos=None, rpy=None, pose=None, color=None,
                 visible=True, transparency=None, name=None):
        if not kinbody.GetName():
            if name is None:
                name = "%s%s" % (type(self).__name__, Body.count)
                Body.count += 1
            kinbody.SetName(name)
        self.rave = kinbody
        self.is_visible = visible
        if pos is not None:
            self.set_pos(pos)
        if rpy is not None:
            self.set_rpy(rpy)
        if pose is not None:
            self.set_pose(pose)
        if color is not None:
            self.set_color(color)
        if not visible:
            self.hide()
        if transparency is not None:
            self.set_transparency(transparency)

    def __str__(self):
        return "pymanoid.Body('%s')" % self.name

    def set_color(self, color):
        """
        Set the color of the rigid body.

        Parameters
        ----------
        color : tuple or string
            RGB tuple, or color code in `matplotlib convention
            <http://matplotlib.org/api/colors_api.html>`_.
        """
        if type(color) is str:
            color = matplotlib_to_rgb(color)
        for link in self.rave.GetLinks():
            for g in link.GetGeometries():
                g.SetAmbientColor(color)
                g.SetDiffuseColor(color)

    def set_transparency(self, transparency):
        """
        Set the transparency of the rigid body.

        Parameters
        ----------
        transparency : double, optional
            Transparency value from 0 (opaque) to 1 (invisible).
        """
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    def show(self):
        """Make the body visible."""
        self.is_visible = True
        self.rave.SetVisible(True)

    def hide(self):
        """Make the body invisible."""
        self.is_visible = False
        self.rave.SetVisible(False)

    @property
    def index(self):
        """
        OpenRAVE index of the body.

        Notes
        -----
        This index is notably used to compute jacobians and hessians.
        """
        return self.rave.GetIndex()

    @property
    def name(self):
        """Body name."""
        return str(self.rave.GetName())

    @property
    def T(self):
        """
        Homogeneous coordinates of the rigid body.

        These coordinates describe the orientation and position of the rigid
        body by the 4 x 4 transformation matrix

        .. math::

            T = \\left[
                \\begin{array}{cc}
                    R & p \\\\
                    0_{1 \\times 3} & 1
                \\end{array}
                \\right]

        where `R` is a `3 x 3` rotation matrix and `p` is the vector of position
        coordinates.

        Notes
        -----
        More precisely, `T` is the transformation matrix *from* the body frame
        *to* the world frame: if :math:`\\tilde{p}_\\mathrm{body} = [x y z 1]`
        denotes the homogeneous coordinates of a point in the body frame, then
        the homogeneous coordinates of this point in the world frame are
        :math:`\\tilde{p}_\\mathrm{world} = T \\tilde{p}_\\mathrm{body}`.
        """
        return self.rave.GetTransform()

    @property
    def pose(self):
        """
        Body pose as a 7D quaternion + position vector.

        The pose vector :math:`[q_w\\,q_x\\,q_y\\,q_z\\,x\\,y\\,z]` consists of
        a quaternion :math:`q = [q_w\\,q_x\\,q_y\\,q_z]` (with the real term
        :math:`q_w` coming first) for the body orientation, followed by the
        coordinates :math:`p = [x\\,y\\,z]` in the world frame.
        """
        pose = self.rave.GetTransformPose()
        if pose[0] < 0:  # convention: cos(alpha) > 0
            # this convention enforces Slerp shortest path
            pose[:4] *= -1
        return pose

    @property
    def R(self):
        """Rotation matrix `R` from local to world coordinates."""
        return self.T[0:3, 0:3]

    @property
    def p(self):
        """Position coordinates `p = [x y z]` in the world frame."""
        return self.T[0:3, 3]

    @property
    def x(self):
        """`x`-coordinate in the world frame."""
        return self.p[0]

    @property
    def y(self):
        """`y`-coordinate in the world frame."""
        return self.p[1]

    @property
    def z(self):
        """`z`-coordinate in the world frame."""
        return self.p[2]

    @property
    def t(self):
        """Tangent vector directing the `x`-axis of the body frame."""
        return self.T[0:3, 0]

    @property
    def b(self):
        """Binormal vector directing the `y`-axis of the body frame."""
        return self.T[0:3, 1]

    @property
    def n(self):
        """Normal vector directing the `z`-axis of the body frame."""
        return self.T[0:3, 2]

    @property
    def quat(self):
        """Quaternion of the rigid body orientation."""
        return self.pose[0:4]

    @property
    def rpy(self):
        """
        Roll-pitch-yaw angles.

        They correspond to Euleur angles for the sequence (1, 2, 3). See
        [Die+06]_ for details.
        """
        return rpy_from_quat(self.quat)

    @property
    def roll(self):
        """Roll angle of the body orientation."""
        return self.rpy[0]

    @property
    def pitch(self):
        """Pitch angle of the body orientation."""
        return self.rpy[1]

    @property
    def yaw(self):
        """Yaw angle of the body orientation."""
        return self.rpy[2]

    def set_transform(self, T):
        """
        Set homogeneous coordinates of the rigid body.

        Parameters
        ----------
        T : array, shape=(4, 4)
            Transform matrix.
        """
        self.rave.SetTransform(T)

    def set_pos(self, pos):
        """
        Set the position of the body in the world frame.

        Parameters
        ----------
        pos : array, shape=(3,)
            3D vector of position coordinates.
        """
        T = self.T.copy()
        T[:3, 3] = pos
        self.set_transform(T)

    def set_rotation_matrix(self, R):
        """
        Set the orientation of the rigid body.

        Recall that this orientation is described by the rotation matrix `R`
        *from* the body frame *to* the world frame.

        Parameters
        ----------
        R : array, shape=(3, 3)
            Rotation matrix.
        """
        T = self.T.copy()
        T[:3, :3] = R
        self.set_transform(T)

    def set_x(self, x):
        """
        Set the `x`-coordinate of the body in the world frame.

        Parameters
        ----------
        x : scalar
            New `x`-coordinate.
        """
        T = self.T.copy()
        T[0, 3] = x
        self.set_transform(T)

    def set_y(self, y):
        """
        Set the `y`-coordinate of the body in the world frame.

        Parameters
        ----------
        y : scalar
            New `y`-coordinate.
        """
        T = self.T.copy()
        T[1, 3] = y
        self.set_transform(T)

    def set_z(self, z):
        """
        Set the `z`-coordinate of the body in the world frame.

        Parameters
        ----------
        z : scalar
            New `z`-coordinate.
        """
        T = self.T.copy()
        T[2, 3] = z
        self.set_transform(T)

    def set_rpy(self, rpy):
        """
        Set the roll-pitch-yaw angles of the body orientation.

        Parameters
        ----------
        rpy : scalar triplet
            Triplet `(r, p, y)` of roll-pitch-yaw angles.
        """
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

    def set_quat(self, quat):
        pose = self.pose.copy()
        pose[0:4] = quat
        self.set_pose(pose)

    def remove(self):
        """
        Remove body from OpenRAVE environment.
        """
        env = get_openrave_env()
        with env:
            env.Remove(self.rave)

    def __del__(self):
        """
        Add body removal to garbage collection step (effective).
        """
        self.remove()

    def apply_twist(self, v, omega, dt):
        """
        Apply a twist [v, omega] defined in the local coordinate frame.

        Parameters
        ----------
        v : array, shape=(3,)
            Linear velocity in local frame.
        omega : array, shape=(3,)
            Angular velocity in local frame.
        dt : scalar
            Duration of twist application in [s].
        """
        self.set_pos(self.p + v * dt)
        self.set_rotation_matrix(self.R + dot(crossmat(omega), self.R) * dt)


class Manipulator(Body):

    """
    Manipulators are special bodies with an end-effector property.

    Parameters
    ----------
    manipulator : openravepy.KinBody
        OpenRAVE manipulator object.
    pos : array, shape=(3,), optional
        Initial position in inertial frame.
    rpy : array, shape=(3,), optional
        Initial orientation in inertial frame.
    pose : array, shape=(7,), optional
        Initial pose. Supersedes ``pos`` and ``rpy`` if they are provided at
        the same time.
    color : char, optional
        Color code in `Matplotlib convention
        <http://matplotlib.org/api/colors_api.html>`_.
    visible : bool, optional
        Initial visibility.
    transparency : double, optional
        Transparency value from 0 (opaque) to 1 (invisible).
    shape : (scalar, scalar)
        Dimensions (half-length, half-width) of a contact patch in [m].
    friction : scalar
        Static friction coefficient for potential contacts.
    """

    def __init__(self, rave_manipulator, pos=None, rpy=None, pose=None,
                 color=None, visible=True, transparency=None, shape=None,
                 friction=None):
        super(Manipulator, self).__init__(
            rave_manipulator, color=color, pos=pos, rpy=rpy, pose=pose,
            visible=visible)
        self.end_effector = rave_manipulator.GetEndEffector()
        self.friction = friction
        self.shape = shape

    def get_contact(self, visible=True, color='r'):
        from contact import Contact
        return Contact(
            self.shape, pose=self.pose, friction=self.friction, visible=visible,
            color=color, link=self)

    @property
    def index(self):
        """
        Index used in Jacobian and Hessian computations.
        """
        return self.end_effector.GetIndex()


class Box(Body):

    """
    Rectangular box.

    Parameters
    ----------
    X : scalar
        Box half-length in [m].
    Y : scalar
        Box half-width in [m].
    Z : scalar
        Box half-height in [m].
    pos : array, shape=(3,)
        Initial position in the world frame.
    rpy : array, shape=(3,)
        Initial orientation in the world frame.
    pose : array, shape=(7,)
        Initial pose in the world frame.
    color : char
        Color letter in ['r', 'g', 'b'].
    visible : bool
        Initial visibility.
    transparency : scalar, optional
        Transparency value from 0 for opaque to 1 for invisible.
    name : string, optional
        Name of the object in the OpenRAVE GUI.
    dZ : scalar, optional
        Shift in box normal coordinates used to make Contact slabs.
    """

    def __init__(self, X, Y, Z, pos=None, rpy=None, pose=None, color='r',
                 visible=True, transparency=None, name=None, dZ=0.):
        self.X = X
        self.Y = Y
        self.Z = Z
        aabb = [0., 0., dZ, X, Y, Z]
        env = get_openrave_env()
        with env:
            box = openravepy.RaveCreateKinBody(env, '')
            box.InitFromBoxes(array([array(aabb)]), True)
            super(Box, self).__init__(
                box, pos=pos, rpy=rpy, pose=pose, color=color, visible=visible,
                transparency=transparency, name=name)
            env.Add(box, True)


class Cube(Box):

    """
    Cube.

    Parameters
    ----------
    size : scalar
        Half-length of a side of the cube in [m].
    pos : array, shape=(3,)
        Initial position in the world frame.
    rpy : array, shape=(3,)
        Initial orientation in the world frame.
    pose : array, shape=(7,)
        Initial pose in the world frame.
    color : char
        Color letter in ['r', 'g', 'b'].
    visible : bool
        Initial visibility.
    transparency : scalar, optional
        Transparency value from 0 for opaque to 1 for invisible.
    name : string, optional
        Name of the object in the OpenRAVE GUI.
    """

    def __init__(self, size, pos=None, rpy=None, pose=None, color='r',
                 visible=True, transparency=None, name=None):
        super(Cube, self).__init__(
            size, size, size, pos=pos, rpy=rpy, color=color, name=name,
            pose=pose, visible=visible, transparency=transparency)


class Point(Cube):

    """
    Points represented by cubes with a default size.

    Parameters
    ----------
    pos : array, shape=(3,)
        Initial position in the world frame.
    size : scalar, optional
        Half-length of a side of the cube in [m].
    color : char
        Color letter in ['r', 'g', 'b'].
    visible : bool
        Initial visibility.
    transparency : scalar, optional
        Transparency value from 0 for opaque to 1 for invisible.
    name : string, optional
        Name of the object in the OpenRAVE GUI.
    """

    def __init__(self, pos, size=0.01, color='r', visible=True,
                 transparency=None, name=None):
        super(Point, self).__init__(
            size, pos=pos, color=color, visible=visible,
            transparency=transparency, name=name)
        self.__pd = zeros(3)

    @property
    def pd(self):
        """Point velocity."""
        return self.__pd.copy()

    def dist(self, other_point):
        """
        Distance to another point.

        Parameters
        ----------
        other_point : array or Point
            Point to compute the distance to.
        """
        if type(other_point) is list:
            other_point = array(other_point)
        if type(other_point) is ndarray:
            return norm(other_point - self.p)
        return norm(other_point.p - self.p)

    def set_vel(self, pd):
        """
        Update the point velocity.

        Parameters
        ----------
        pd : array, shape=(3,)
            Velocity coordinates in the world frame.
        """
        self.__pd = array(pd)

    def integrate_acceleration(self, pdd, dt):
        """
        Euler integration of constant acceleration ``pdd`` over duration ``dt``.

        Parameters
        ----------
        pdd : array, shape=(3,)
            Acceleration coordinates in the world frame.
        dt : scalar
            Duration in [s].
        """
        self.set_pos(self.p + (self.pd + .5 * pdd * dt) * dt)
        self.set_vel(self.pd + pdd * dt)


class PointMass(Point):

    """
    Point with a mass property and a size proportional to it.

    Parameters
    ----------
    pos : array, shape=(3,)
        Initial position in the world frame.
    mass : scalar
        Total mass in [kg].
    color : char
        Color letter in ['r', 'g', 'b'].
    visible : bool
        Initial visibility.
    transparency : scalar, optional
        Transparency value from 0 for opaque to 1 for invisible.
    name : string, optional
        Name of the object in the OpenRAVE GUI.
    """
    def __init__(self, pos, mass, color='r', visible=True, transparency=None,
                 name=None):
        size = max(5e-3, 6e-4 * mass)
        super(PointMass, self).__init__(
            pos, size, color=color, visible=visible, transparency=transparency,
            name=name)
        self.mass = mass
