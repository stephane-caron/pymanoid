#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Hervé Audren <herve.audren@lirmm.fr>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with pymanoid. If not, see <http://www.gnu.org/licenses/>.

"""
This example shows a contact-stability condition: the robust static-equilibrium
CoM polyhedron. See <https://hal-lirmm.ccsd.cnrs.fr/lirmm-01477362> for details.
Running this example requires the StabiliPy
<https://github.com/haudren/stabilipy> library.
"""

import IPython

from numpy import zeros

import pymanoid

from pymanoid import Stance
from pymanoid.gui import PointMassWrenchDrawer
from pymanoid.gui import draw_polyhedron
from pymanoid.misc import matplotlib_to_rgb, norm

from openravepy import matrixFromPose
import numpy as np

import stabilipy as stab

mass = 50.


class SupportPolyhedronDrawer(pymanoid.Process):

    """
    Draw the robust static-equilibrium polyhedron of a contact set.

    Parameters
    ----------
    stance : Stance
        Contacts and COM position of the robot.
    color : tuple or string, optional
        Area color.
    method: string, optional
        Method to compute the static equilibrium polygon.
        Choices are cdd, qhull (default) and parma
    """

    def __init__(self, stance, z=0., color=None, method='qhull'):
        if color is None:
            color = (0., 0.5, 0., 0.5)
        if type(color) is str:
            color = matplotlib_to_rgb(color) + [0.5]
        super(SupportPolyhedronDrawer, self).__init__()
        self.color = color
        self.contact_poses = {}
        self.polyhedron = None
        self.handle = None
        self.stance = stance
        self.z = z
        self.method = method
        #
        self.update_contacts()
        self.create_polyhedron(self.stance.contacts)

        self.nr_iter = 0
        self.max_iter = 50

    def clear(self):
        self.handle = None

    def on_tick(self, sim):
        if self.handle is None:
            self.create_polyhedron(self.stance.contacts)
            return
        for contact in self.stance.contacts:
            if norm(contact.pose - self.contact_poses[contact.name]) > 1e-10:
                self.update_contacts()
                self.create_polyhedron(self.stance.contacts)
                return
        if self.nr_iter < self.max_iter:
            self.refine_polyhedron()
            self.nr_iter += 1

    def update_contacts(self):
        for contact in self.stance.contacts:
            self.contact_poses[contact.name] = contact.pose

    def create_polyhedron(self, contacts):
        self.handle = None
        self.nr_iter = 0
        try:
            self.polyhedron = stab.StabilityPolygon(mass, dimension=3, radius=1.5)
            stab_contacts = []
            for contact in contacts:
                hmatrix = matrixFromPose(contact.pose)
                X, Y = contact.shape
                displacements = [ np.array([[X, Y, 0]]).T,
                                  np.array([[-X, Y, 0]]).T,
                                  np.array([[-X, -Y, 0]]).T,
                                  np.array([[X, -Y, 0]]).T
                                  ]
                for displacement in displacements:
                    stab_contacts.append(stab.Contact(contact.friction,
                                                      hmatrix[:3, 3:]+hmatrix[:3, :3].dot(displacement),
                                                 hmatrix[:3, 2:3]))
            self.polyhedron.contacts = stab_contacts

            self.polyhedron.select_solver(self.method)
            self.polyhedron.make_problem()
            self.polyhedron.init_algo()
            self.polyhedron.build_polys()

            vertices = self.polyhedron.polyhedron()
            self.handle = draw_polyhedron(
                [(x[0], x[1], x[2]) for x in vertices])
        except Exception as e:
            print "SupportPolyhedronDrawer:", e

    def refine_polyhedron(self):
        try:
            self.polyhedron.next_edge()
            vertices = self.polyhedron.polyhedron()
            self.handle = draw_polyhedron(
                [(x[0], x[1], x[2]) for x in vertices])
        except Exception as e:
            print "SupportPolyhedronDrawer:", e

    def update_z(self, z):
        self.z = z
        self.update_polygon()


class StaticWrenchDrawer(PointMassWrenchDrawer):

    """
    Draw contact wrenches applied to a robot in static-equilibrium.

    Parameters
    ----------
    stance : Stance
        Contacts and COM position of the robot.
    """

    def __init__(self, stance):
        super(StaticWrenchDrawer, self).__init__(stance.com, stance)
        stance.com.pdd = zeros((3,))
        self.stance = stance

    def find_supporting_wrenches(self, sim):
        return self.stance.find_static_supporting_wrenches()


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

    stance = Stance.from_json('../stances/double.json')
    stance.bind(robot)
    robot.ik.solve()

    polygon_drawer = SupportPolyhedronDrawer(stance)
    wrench_drawer = StaticWrenchDrawer(stance)

    sim.schedule(robot.ik)
    sim.schedule_extra(polygon_drawer)
    sim.schedule_extra(wrench_drawer)
    sim.start()

    print """
COM robust static-equilibrium polygon
==============================

Ready to go! The GUI displays the COM static-equilibrium polygon in green. You
can move the blue box (in the plane above the robot) around to make the robot
move its center of mass. Contact wrenches are displayed at each contact (green
dot is COP location, arrow is resultant force). When the COM exits the
static-equilibrium polygon, you should see the background turn red as no
feasible contact wrenches can be found.

Enjoy :)
"""

    if IPython.get_ipython() is None:
        IPython.embed()
