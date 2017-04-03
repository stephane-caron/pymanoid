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

import IPython

try:
    import pymanoid
except ImportError:
    import os
    import sys
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../')
    import pymanoid

from pymanoid import PointMass, Stance
from pymanoid.contact import Contact
from pymanoid.drawers import SEPDrawer, StaticWrenchDrawer

K_doflim = 5.
com_height = 0.9  # [m]
polygon_handle = None
qd_lim = 10.
z_polygon = 2.


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

    com_target = PointMass(
        pos=[0., 0., com_height], mass=robot.mass, color='b', visible=False)
    com_above = pymanoid.Cube(0.02, [0.05, 0.04, z_polygon], color='b')

    stance = Stance(
        com=com_target,
        left_foot=Contact(
            shape=robot.sole_shape,
            pos=[0.20, 0.15, 0.1],
            rpy=[0, 0, 0],
            friction=0.5,
            visible=True),
        right_foot=Contact(
            shape=robot.sole_shape,
            pos=[-0.2, -0.195, 0.],
            rpy=[0, 0, 0],
            friction=0.5,
            visible=True),
        left_hand=Contact(
            shape=robot.sole_shape,
            pos=[0.45, 0.46, 0.96],
            rpy=[0., -0.8, 0.5],
            friction=0.5,
            visible=True))

    robot.init_ik(active_dofs=robot.whole_body)
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
    robot.generate_posture(stance, debug=True)

    print ""
    print "In this example, we display the static-equilibrium COM polygon"
    print "(in magenta) for a given set of contacts."
    print ""
    print "You can move contacts by selecting them in the OpenRAVE GUI."
    print "The robot IK is servoed to their positions."
    print ""
    print "To illustrate the validity of this polygon, contact forces are"
    print "computed that support the equilibrium position represented by"
    print "the blue box (which acts like a COM position). Try moving this"
    print "box around, and see what happens when it exits the polygon."
    print ""

    wrench_drawer = StaticWrenchDrawer(com_target, stance)
    sep_drawer = SEPDrawer(stance, z_polygon)

    sim.schedule(robot.ik)
    sim.schedule(wrench_drawer)
    sim.schedule(sep_drawer)
    sim.schedule(COMSync())
    sim.start()

    if IPython.get_ipython() is None:
        IPython.embed()
