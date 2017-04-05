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

from numpy import hstack, zeros
from time import time

from body import PointMass
from draw import draw_line, draw_polygon, draw_polyhedron
from draw import draw_wrench
from misc import norm
from sim import Process


class WrenchDrawer(Process):

    """
    Draw contact wrenches applied to the robot.

    Parameters
    ----------
    scale : scalar
        Force-to-distance conversion ratio in [m] / [N].
    """

    DEFAULT_SCALE = 0.0025
    KO_COLOR = [.8, .4, .4]

    def __init__(self, scale=None):
        super(WrenchDrawer, self).__init__()
        scale = self.DEFAULT_SCALE if scale is None else scale
        self.handles = []
        self.last_bkgnd_switch = None
        self.nb_fails = 0
        self.scale = scale

    def clear(self):
        self.handles = []

    def find_supporting_wrenches(self, sim):
        raise NotImplementedError("should be implemented by child classes")

    def on_tick(self, sim):
        """Find supporting contact forces at each COM acceleration update."""
        try:
            support = self.find_supporting_wrenches(sim)
            self.handles = [
                draw_wrench(contact, w_c, scale=self.scale)
                for (contact, w_c) in support]
        except ValueError:
            self.handles = []
            self.nb_fails += 1
            sim.viewer.SetBkgndColor(self.KO_COLOR)
            self.last_bkgnd_switch = time()
        if self.last_bkgnd_switch is not None \
                and time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            sim.viewer.SetBkgndColor(sim.BACKGROUND_COLOR)
            self.last_bkgnd_switch = None


class PointMassWrenchDrawer(WrenchDrawer):

    """
    Draw contact wrenches applied to a point-mass system in multi-contact.

    Parameters
    ----------
    point_mass : PointMass
        Point-mass to which forces are applied.
    contact_set : ContactSet
        Set of contacts providing interaction forces.
    scale : scalar
        Force-to-distance conversion ratio in [m] / [N].
    """

    def __init__(self, point_mass, contact_set, scale=None):
        super(PointMassWrenchDrawer, self).__init__(scale)
        self.contact_set = contact_set
        self.point_mass = point_mass

    def find_supporting_wrenches(self, sim):
        mass = self.point_mass.mass
        p = self.point_mass.p
        pdd = self.point_mass.pdd
        wrench = hstack([mass * (pdd - sim.gravity), zeros(3)])
        support = self.contact_set.find_supporting_wrenches(wrench, p)
        return support

    def on_tick(self, sim):
        if self.point_mass.pdd is None:  # needs to be stored by the user
            return
        super(PointMassWrenchDrawer, self).on_tick(sim)


class RobotWrenchDrawer(WrenchDrawer):

    """
    Draw contact wrenches applied to a humanoid in multi-contact.

    Parameters
    ----------
    robot : Humanoid
        Humanoid robot model.
    scale : scalar
        Force-to-distance conversion ratio in [m] / [N].
    """

    def __init__(self, robot, scale=None):
        super(RobotWrenchDrawer, self).__init__(scale)
        self.robot = robot
        self.qd_prev = robot.qd

    def find_supporting_wrenches(self, sim):
        qd = self.robot.qd
        qdd = (qd - self.qd_prev) / sim.dt
        self.qd_prev = qd
        com = self.robot.com
        stance = self.robot.stance
        contact_wrench = -self.robot.compute_gravito_inertial_wrench(qdd, com)
        support = stance.find_supporting_wrenches(contact_wrench, com)
        return support


class StaticWrenchDrawer(PointMassWrenchDrawer):

    """
    Draw contact wrenches applied to a robot in static-equilibrium.

    Parameters
    ----------
    point_mass : PointMass
        Point-mass to which forces are applied.
    contact_set : ContactSet
        Set of contacts providing interaction forces.
    scale : scalar
        Force-to-distance conversion ratio in [m] / [N].
    """

    def __init__(self, point_mass, contact_set, scale=0.0025):
        super(StaticWrenchDrawer, self).__init__(point_mass, contact_set, scale)
        self.point_mass.pdd = zeros((3,))

    def find_supporting_wrenches(self, sim):
        mass = self.point_mass.mass
        p = self.point_mass.p
        support = self.contact_set.find_static_supporting_wrenches(p, mass)
        return support


"""
Trajectory drawers
==================
"""


class TrajectoryDrawer(Process):

    """
    Draw the trajectory of a rigid body.

    Parameters
    ----------
    body : Body
        Rigid body whose trajectory to draw.
    combined : string, optional
        Drawing spec of the trajectory in matplotlib fashion.
    color : char or RGBA tuple, optional
        Drawing color.
    linewidth : scalar, optional
        Thickness of drawn lines.
    linestyle : char, optional
        Choix between '-' for continuous and '.' for dotted.
    buffer_size : int, optional
        Number of trajectory segments to display. Old segments will be replaced
        by new ones.
    """

    def __init__(self, body, combined='b-', color=None, linewidth=3,
                 linestyle=None, buffer_size=1000):
        super(TrajectoryDrawer, self).__init__()
        color = color if color is not None else combined[0]
        linestyle = linestyle if linestyle is not None else combined[1]
        assert linestyle in ['-', '.']
        self.body = body
        self.buffer_size = buffer_size
        self.color = color
        self.handles = [None] * buffer_size
        self.next_index = 0
        self.last_pos = body.p
        self.linestyle = linestyle
        self.linewidth = linewidth

    def on_tick(self, sim):
        if self.linestyle == '-':
            h = self.handles[self.next_index]
            if h is not None:
                h.Close()
            self.handles[self.next_index] = draw_line(
                self.last_pos, self.body.p, color=self.color,
                linewidth=self.linewidth)
            self.next_index = (self.next_index + 1) % self.buffer_size
        self.last_pos = self.body.p

    def dash_graph_handles(self):
        for i in xrange(len(self.handles)):
            if i % 2 == 0:
                self.handles[i] = None


class COMTrajectoryDrawer(TrajectoryDrawer):

    """
    Draw the COM trajectory of a robot.

    Parameters
    ----------
    robot : Robot
        Robot model to track.
    combined : string, optional
        Drawing spec of the trajectory in matplotlib fashion.
    color : char or RGBA tuple
        Drawing color.
    linewidth : scalar, optional
        Thickness of drawn lines.
    linestyle : char, optional
        Choix between '-' for continuous and '.' for dotted.
    buffer_size : int, optional
        Number of trajectory segments to display. Old segments will be replaced
        by new ones.
    """

    def __init__(self, robot, combined='b-', color=None, linewidth=3,
                 linestyle=None, buffer_size=1000):
        body = PointMass(
            robot.com, robot.mass, name='RobotCOMState', visible=False)
        super(COMTrajectoryDrawer, self).__init__(
            body, combined, color, linewidth, linestyle)
        self.robot = robot

    def on_tick(self, sim):
        self.body.set_pos(self.robot.com)
        super(COMTrajectoryDrawer, self).on_tick(sim)


"""
Support areas and volumes
=========================
"""


class SupportAreaDrawer(Process):

    """
    Draw a given support area of a contact set.

    Parameters
    ----------
    contact_set : ContactSet
        Contact set to track.
    z : scalar, optional
        Altitude of drawn area in the world frame.
    """

    def __init__(self, contact_set, z=0.):
        super(SupportAreaDrawer, self).__init__()
        self.contact_poses = {}
        self.contact_set = contact_set
        self.handle = None
        self.z = z
        self.update_contact_poses()
        self.update_polygon()

    def on_tick(self, sim):
        for contact in self.contact_set.contacts:
            if norm(contact.pose - self.contact_poses[contact.name]) > 1e-10:
                self.update_contact_poses()
                self.update_polygon()
                break

    def update_contact_poses(self):
        for contact in self.contact_set.contacts:
            self.contact_poses[contact.name] = contact.pose

    def update_polygon(self):
        raise NotImplementedError


class SEPDrawer(SupportAreaDrawer):

    """
    Draw the static-equilibrium polygon of a contact set.

    Parameters
    ----------
    contact_set : ContactSet
        Contact set to track.
    z : scalar, optional
        Altitude of drawn area in the world frame.
    """

    def update_polygon(self):
        self.handle = None
        try:
            vertices = self.contact_set.compute_static_equilibrium_polygon()
            self.handle = draw_polygon(
                [(x[0], x[1], self.z) for x in vertices],
                normal=[0, 0, 1], color=(0.5, 0., 0.5, 0.5))
        except Exception as e:
            print "SEPDrawer:", e


class ZMPSupportAreaDrawer(SupportAreaDrawer):

    """
    Draw the pendular ZMP area of a contact set.

    Parameters
    ----------
    contact_set : ContactSet
        Contact set to track.
    z : scalar, optional
        Altitude of drawn area in the world frame.
    """

    def __init__(self, stance, z=0.):
        self.last_com = stance.com.p
        self.method = 'cdd'
        self.stance = stance
        super(ZMPSupportAreaDrawer, self).__init__(stance, z)

    def on_tick(self, sim):
        super(ZMPSupportAreaDrawer, self).on_tick(sim)
        if norm(self.stance.com.p - self.last_com) > 1e-10:
            self.update_contact_poses()
            self.update_polygon()
            self.last_com = self.stance.com.p

    def update_polygon(self):
        self.handle = None
        try:
            vertices = self.contact_set.compute_zmp_support_area(
                self.stance.com.p, [0, 0, self.z], method=self.method)
            self.handle = draw_polygon(
                [(x[0], x[1], self.z) for x in vertices],
                normal=[0, 0, 1], color=(0.0, 0.5, 0.5, 0.5))
        except Exception as e:
            print "ZMPSupportAreaDrawer:", e


class COMAccelConeDrawer(ZMPSupportAreaDrawer):

    """
    Draw the COM acceleration cone of a contact set.

    Parameters
    ----------
    contact_set : ContactSet
        Contact set to track.
    """

    def __init__(self, contact_set):
        super(COMAccelConeDrawer, self).__init__(contact_set)

    def update_polygon(self):
        self.handle = None
        try:
            cone_vertices = self.contact_set.compute_pendular_accel_cone(
                self.stance.com.p)
            vscale = [self.stance.com.p + 0.1 * acc for acc in cone_vertices]
            self.handle = draw_polyhedron(vscale, 'r.-#')
        except Exception as e:
            print "COMAccelConeDrawer:", e
