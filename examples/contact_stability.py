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

"""
This example shows three contact-stability conditions: the static-equilibrium
COM polygon, the dynamic ZMP support area, and the 3D COM acceleration cone.
See <https://scaron.info/research/tro-2016.html> for details.
"""

import IPython

from numpy import zeros

import pymanoid

from pymanoid import PointMass, Stance
from pymanoid.contact import Contact
from pymanoid.gui import PointMassWrenchDrawer
from pymanoid.gui import draw_polygon, draw_polyhedron
from pymanoid.misc import matplotlib_to_rgb, norm

com_height = 0.9  # [m]
z_polygon = 2.


class SupportAreaDrawer(pymanoid.Process):

    """
    Draw a given support area of a contact set.

    Parameters
    ----------
    contact_set : ContactSet, optional
        Contact set to track.
    z : scalar, optional
        Altitude of drawn area in the world frame.
    color : tuple or string, optional
        Area color.
    """

    def __init__(self, contact_set=None, z=0., color=None):
        if color is None:
            color = (0., 0.5, 0., 0.5)
        if type(color) is str:
            color = matplotlib_to_rgb(color) + [0.5]
        super(SupportAreaDrawer, self).__init__()
        self.color = color
        self.contact_poses = {}
        self.contact_set = contact_set
        self.handle = None
        self.z = z
        if contact_set is not None:
            self.update_contact_poses()
            self.update_polygon()

    def clear(self):
        self.handle = None

    def on_tick(self, sim):
        if self.handle is None:
            self.update_polygon()
        for contact in self.contact_set.contacts:
            if norm(contact.pose - self.contact_poses[contact.name]) > 1e-10:
                self.update_contact_poses()
                self.update_polygon()
                break

    def update_contact_poses(self):
        for contact in self.contact_set.contacts:
            self.contact_poses[contact.name] = contact.pose

    def update_contact_set(self, contact_set):
        self.contact_set = contact_set
        self.update_contact_poses()
        self.update_polygon()

    def update_polygon(self):
        raise NotImplementedError

    def update_z(self, z):
        self.z = z
        self.update_polygon()


class SEPDrawer(SupportAreaDrawer):

    """
    Draw the static-equilibrium polygon of a contact set.

    Parameters
    ----------
    contact_set : ContactSet, optional
        Contact set to track.
    z : scalar, optional
        Altitude of drawn area in the world frame.
    color : tuple or string, optional
        Area color.
    """

    def update_polygon(self):
        self.handle = None
        try:
            vertices = self.contact_set.compute_static_equilibrium_polygon()
            self.handle = draw_polygon(
                [(x[0], x[1], self.z) for x in vertices],
                normal=[0, 0, 1], color=self.color)
        except Exception as e:
            print "SEPDrawer:", e


class ZMPSupportAreaDrawer(SupportAreaDrawer):

    """
    Draw the pendular ZMP area of a contact set.

    Parameters
    ----------
    stance : Stance
        Stance to track.
    z : scalar, optional
        Altitude of drawn area in the world frame.
    color : tuple or string, optional
        Area color.
    """

    def __init__(self, stance, z=0., color=None):
        self.stance = stance  # before calling parent constructor
        super(ZMPSupportAreaDrawer, self).__init__(stance, z, color)
        self.last_com = stance.com.p

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
                self.stance.com.p, [0, 0, self.z])
            self.handle = draw_polygon(
                [(x[0], x[1], self.z) for x in vertices],
                normal=[0, 0, 1], color=(0.0, 0.0, 0.5, 0.5))
        except Exception as e:
            print "ZMPSupportAreaDrawer:", e


class COMAccelConeDrawer(ZMPSupportAreaDrawer):

    """
    Draw the COM acceleration cone of a contact set.

    Parameters
    ----------
    stance : Stance
        Contact set to track.
    scale : scalar, optional
        Acceleration to distance conversion ratio, in [s]^2.
    color : tuple or string, optional
        Area color.
    """

    def __init__(self, stance, scale=0.1, color=None):
        self.scale = scale  # done before calling parent constructor
        super(COMAccelConeDrawer, self).__init__(stance, color=color)

    def update_polygon(self):
        self.handle = None
        try:
            vertices = self.contact_set.compute_pendular_accel_cone(
                self.stance.com.p)
            vscale = [self.stance.com.p + self.scale * acc for acc in vertices]
            self.handle = draw_polyhedron(vscale, 'r.-#')
        except Exception as e:
            print "COMAccelConeDrawer:", e


class StaticWrenchDrawer(PointMassWrenchDrawer):

    """
    Draw contact wrenches applied to a robot in static-equilibrium.

    Parameters
    ----------
    point_mass : PointMass
        Point-mass to which forces are applied.
    contact_set : ContactSet
        Set of contacts providing interaction forces.
    """

    def __init__(self, point_mass, contact_set):
        super(StaticWrenchDrawer, self).__init__(point_mass, contact_set)
        self.point_mass.pdd = zeros((3,))

    def find_supporting_wrenches(self, sim):
        mass = self.point_mass.mass
        p = self.point_mass.p
        support = self.contact_set.find_static_supporting_wrenches(p, mass)
        return support


class COMSync(pymanoid.Process):

    def on_tick(self, sim):
        com_target.set_x(com_above.x)
        com_target.set_y(com_above.y)


if __name__ == "__main__":
    sim = pymanoid.Simulation(dt=0.03)
    robot = pymanoid.robots.JVRC1('JVRC-1.dae', download_if_needed=True)
    sim.set_viewer()
    sim.viewer.SetCamera([
        [0.60587192, -0.36596244,  0.70639274, -2.4904027],
        [-0.79126787, -0.36933163,  0.48732874, -1.6965636],
        [0.08254916, -0.85420468, -0.51334199,  2.79584694],
        [0.,  0.,  0.,  1.]])
    robot.set_transparency(0.25)
    robot.set_dof_values([
        3.53863816e-02,   2.57657518e-02,   7.75586039e-02,
        6.35909636e-01,   7.38580762e-02,  -5.34226902e-01,
        -7.91656626e-01,   1.64846093e-01,  -2.13252247e-01,
        1.12500819e+00,  -1.91496369e-01,  -2.06646315e-01,
        1.39579597e-01,  -1.33333598e-01,  -8.72664626e-01,
        0.00000000e+00,  -9.81307787e-15,   0.00000000e+00,
        -8.66484961e-02,  -1.78097540e-01,  -1.68940240e-03,
        -5.31698601e-01,  -1.00166891e-04,  -6.74394930e-04,
        -1.01552628e-04,  -5.71121132e-15,  -4.18037117e-15,
        0.00000000e+00,   0.00000000e+00,   0.00000000e+00,
        0.00000000e+00,  -7.06534763e-01,   1.67723830e-01,
        2.40289101e-01,  -1.11674923e+00,   6.23384177e-01,
        -8.45611535e-01,   1.39994759e-02,   1.17756934e-16,
        3.14018492e-16,  -3.17943723e-15,  -6.28036983e-16,
        -3.17943723e-15,  -6.28036983e-16,  -6.88979202e-02,
        -4.90099381e-02,   8.17415141e-01,  -8.71841480e-02,
        -1.36966665e-01,  -4.26226421e-02])

    com_target = PointMass(pos=[0., 0., com_height], mass=robot.mass, color='b')
    com_target.hide()
    com_above = pymanoid.Cube(0.02, [0.05, 0.04, z_polygon], color='b')

    stance = Stance(
        com=com_target,
        left_foot=Contact(
            shape=robot.sole_shape,
            pos=[0.20, 0.15, 0.1],
            rpy=[0.4, 0, 0],
            friction=0.5),
        right_foot=Contact(
            shape=robot.sole_shape,
            pos=[-0.2, -0.195, 0.],
            rpy=[-0.4, 0, 0],
            friction=0.5))
    stance.bind(robot)
    robot.ik.solve()

    com_sync = COMSync()
    cone_drawer = COMAccelConeDrawer(stance, scale=0.05)
    sep_drawer = SEPDrawer(stance, z_polygon)
    wrench_drawer = StaticWrenchDrawer(com_target, stance)
    zmp_area_drawer = ZMPSupportAreaDrawer(stance, z_polygon)

    sim.schedule(robot.ik)
    sim.schedule_extra(com_sync)
    sim.schedule_extra(cone_drawer)
    sim.schedule_extra(sep_drawer)
    sim.schedule_extra(wrench_drawer)
    sim.schedule_extra(zmp_area_drawer)
    sim.start()

    print """
Contact-stability conditions
============================

Ready to go! The GUI displays three contact-stability criteria:

    Blue polygon -- ZMP pendular support area
    Green polygon -- COM static-equilibrium polygon
    Red cone -- COM pendular acceleration cone

You can move the blue box (in the plane above the robot) around to make the
robot move its center of mass. Contact wrenches are displayed at each contact
(green dot is COP location, arrow is resultant force).

When the COM exists the static-equilibrium polygon, you should see the
background turn red as no feasible contact wrenches can be found.

Enjoy :)

"""

    if IPython.get_ipython() is None:
        IPython.embed()
