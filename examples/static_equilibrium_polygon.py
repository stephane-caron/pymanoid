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

import IPython
import pymanoid
import thread
import threading
import time


com_height = 0.9  # [m]
dt = 3e-2  # [s]
env_lock = threading.Lock()
polygon_handle = None
z_polygon = 2.

qd_lim = 10.
K_doflim = 5.


def run_forces_thread():
    handles = []
    while True:
        env_lock.acquire()
        com_target.set_x(com_above.x)
        com_target.set_y(com_above.y)
        try:
            support = contacts.find_static_supporting_forces(
                com_target.p, robot.mass)
            handles = [pymanoid.draw_force(c, fc) for (c, fc) in support]
        except Exception as e:
            print "Force computation failed:", e
            print "Did you move the target COM (blue box) out of the polygon?\n"
        env_lock.release()
        time.sleep(dt)
    return handles


def recompute_polygon():
    global polygon_handle
    vertices = contacts.compute_static_equilibrium_area(robot.mass)
    polygon_handle = pymanoid.draw_polygon(
        [(x[0], x[1], z_polygon) for x in vertices],
        normal=[0, 0, 1], color=(0.5, 0., 0.5, 0.5))


if __name__ == "__main__":
    pymanoid.init()
    robot = pymanoid.robots.JVRC1('JVRC-1.dae', download_if_needed=True)
    pymanoid.get_viewer().SetCamera([
        [0.60587192, -0.36596244,  0.70639274, -2.4904027],
        [-0.79126787, -0.36933163,  0.48732874, -1.6965636],
        [0.08254916, -0.85420468, -0.51334199,  2.79584694],
        [0.,  0.,  0.,  1.]])
    robot.set_transparency(0.25)

    contacts = pymanoid.ContactSet({
        'left_foot': pymanoid.Contact(
            X=0.2,
            Y=0.1,
            pos=[0.20, 0.15, 0.1],
            rpy=[0, 0, 0],
            friction=0.5,
            visible=True),
        'right_foot': pymanoid.Contact(
            X=0.2,
            Y=0.1,
            pos=[-0.2, -0.195, 0.],
            rpy=[0, 0, 0],
            friction=0.5,
            visible=True),
        'left_hand': pymanoid.Contact(
            X=0.2,
            Y=0.1,
            pos=[0.45, 0.46, 0.96],
            rpy=[0., -0.8, 0.5],
            friction=0.5,
            visible=True)
    })

    com_target = pymanoid.Cube(0.02, pos=[0., 0., com_height], color='b',
                               visible=False)
    com_above = pymanoid.Cube(0.02, [0.05, 0.04, z_polygon], color='b')

    active_dofs = robot.chest + robot.free + robot.left_arm + \
        robot.right_arm + robot.left_leg + robot.right_leg
    robot.set_active_dofs(active_dofs)
    robot.init_ik(qd_lim=qd_lim, K_doflim=K_doflim)
    # robot.generate_posture(contacts, com_target=[0.05,  0.04,  0.90])
    robot.generate_posture(contacts, com_target)
    recompute_polygon()

    print ""
    print "In this example, we display the static-equilibrium COM polygon"
    print "(in magenta) for a given set of contacts."
    print ""
    print "You can move contacts by selecting them in the OpenRAVE GUI."
    print "The robot IK is servoed to their positions. Type:"
    print ""
    print "    recompute_polygon()"
    print ""
    print "to recompute the COM polygon after moving contacts."
    print ""
    print "To illustrate the validity of this polygon, contact forces are"
    print "computed that support the equilibrium position represented by"
    print "the blue box (which acts like a COM position). Try moving this"
    print "box around, and see what happens when it exits the polygon."
    print ""

    robot.start_ik_thread(dt)
    thread.start_new_thread(run_forces_thread, ())
    IPython.embed()
