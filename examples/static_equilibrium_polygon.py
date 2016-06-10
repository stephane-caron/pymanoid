#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
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


import IPython
import numpy
import openravepy
import os.path
import pymanoid
import thread
import threading
import time


dt = 3e-2  # [s]
env_file = './openrave_models/JVRC-1/env.xml'
env_lock = threading.Lock()
polygon_handle = None
z_polygon = 2.

# IK settings
qd_lim = 10.
K_doflim = 5.
G_com = 1. / dt
G_contact = 0.9 / dt
w_link = 100.
w_com = 005.
w_reg = 001.


def run_ik_thread():
    while True:
        env_lock.acquire()
        robot.step_ik(dt)
        env_lock.release()
        time.sleep(dt)


def run_forces_thread():
    handles = []
    while True:
        env_lock.acquire()
        try:
            support = contacts.find_static_supporting_forces(
                outbox.p, robot.mass)
            handles = [pymanoid.draw_force(env, c, fc) for (c, fc) in support]
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
        env, [(x[0], x[1], outbox.z) for x in vertices],
        n=[0, 0, 1], color=(0.5, 0., 0.5, 0.5))


def generate_posture(robot, contacts, com_target):
    active_dofs = robot.chest_dofs + robot.free_dofs + robot.left_arm_dofs + \
        robot.right_arm_dofs + robot.left_leg_dofs + robot.right_leg_dofs
    robot.set_active_dofs(active_dofs)
    robot.init_ik(qd_lim=qd_lim, K_doflim=K_doflim)
    robot.add_com_objective(com_target, G_com, w_com)
    robot.add_contact_objective(
        robot.left_foot, contacts['left_foot'], G_contact, w_link)
    robot.add_contact_objective(
        robot.right_foot, contacts['right_foot'], G_contact, w_link)
    if 'left_hand' in contacts:
        robot.add_contact_objective(
            robot.left_hand, contacts['left_hand'], G_contact, w_link)
    robot.add_velocity_regularization(w_reg)
    robot.solve_ik(dt, conv_tol=1e-4, max_it=200, debug=False)


if __name__ == "__main__":
    if not os.path.isfile(env_file):
        print "Error opening file %s" % env_file
        print "                                                                "
        print "For this example, you need to clone the models repository:      "
        print "                                                                "
        print "    git clone https://github.com/stephane-caron/openrave_models "
        print "                                                                "
        print "and link it in this folder (or update `env_file` in the script)."
        print "                                                                "
        exit(-1)

    env = openravepy.Environment()
    env.Load(env_file)
    env.SetViewer('qtcoin')
    viewer = env.GetViewer()
    viewer.SetBkgndColor([.6, .6, .8])
    viewer.SetCamera(numpy.array([
        [0.60587192, -0.36596244,  0.70639274, -2.4904027],
        [-0.79126787, -0.36933163,  0.48732874, -1.6965636],
        [0.08254916, -0.85420468, -0.51334199,  2.79584694],
        [0.,  0.,  0.,  1.]]))
    robot = pymanoid.robots.JVRC1(env)
    robot.set_transparency(0.25)

    contacts = pymanoid.ContactSet({
        'left_foot': pymanoid.Contact(
            env,
            X=0.2,
            Y=0.1,
            pos=[0.20, 0.15, 0.1],
            rpy=[0, 0, 0],
            friction=0.5,
            visible=True),
        'right_foot': pymanoid.Contact(
            env,
            X=0.2,
            Y=0.1,
            pos=[-0.2, -0.195, 0.],
            rpy=[0, 0, 0],
            friction=0.5,
            visible=True),
        'left_hand': pymanoid.Contact(
            env,
            X=0.2,
            Y=0.1,
            pos=[0.45, 0.46, 0.96],
            rpy=[0., -0.8, 0.],
            friction=0.5,
            visible=True)
    })

    outbox = pymanoid.Cube(env, 0.02, [0, 0, z_polygon], color='b')
    generate_posture(robot, contacts, com_target=[0.05,  0.04,  0.90])
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

    thread.start_new_thread(run_ik_thread, ())
    thread.start_new_thread(run_forces_thread, ())
    IPython.embed()
