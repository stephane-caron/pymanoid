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
This example goes into the details of the IK solver setup. Beginners should take
a look at the posture-generation example first.
"""

import IPython
import numpy
import time

import pymanoid

from pymanoid import Contact, PointMass
from pymanoid.robots import JVRC1
from pymanoid.tasks import COMTask, ContactTask, DOFTask, PostureTask


def move_com_back_and_forth(duration, dt=1e-2):
    init_com = robot.com.copy()
    for t in numpy.arange(0, duration, dt):
        loop_start = time.time()
        com_var = numpy.sin(t) * numpy.array([.2, 0, 0])
        com.set_pos(init_com + numpy.array([-0.2, 0., 0.]) + com_var)
        robot.ik.step(dt)
        rem_time = dt - (time.time() - loop_start)
        if rem_time > 0:
            time.sleep(rem_time)


if __name__ == '__main__':
    sim = pymanoid.Simulation(dt=0.03)
    robot = JVRC1('JVRC-1.dae', download_if_needed=True)
    sim.set_viewer()
    sim.viewer.SetCamera([
        [-0.28985317,  0.40434422, -0.86746233,  2.73872042],
        [0.95680251,  0.10095043, -0.2726499,  0.86080128],
        [-0.02267371, -0.90901857, -0.41613837,  2.06654644],
        [0.,  0.,  0.,  1.]])

    # IK targets
    lf_target = Contact(robot.sole_shape, pos=[0, 0.3, 0])
    rf_target = Contact(robot.sole_shape, pos=[0, -0.3, 0])

    # Initial robot pose
    robot.set_dof_values([0.8], dof_indices=[robot.TRANS_Z])
    com = PointMass(pos=robot.com, mass=robot.mass)

    # IK tasks
    lf_task = ContactTask(
        robot, robot.left_foot, lf_target, weight=1., gain=0.85)
    rf_task = ContactTask(
        robot, robot.right_foot, rf_target, weight=1., gain=0.85)
    com_task = COMTask(robot, com, weight=1e-2, gain=0.85)
    reg_task = PostureTask(robot, robot.q, weight=1e-6, gain=0.85)

    # IK setup
    robot.ik.clear_tasks()
    robot.ik.add_task(lf_task)
    robot.ik.add_task(rf_task)
    robot.ik.add_task(com_task)
    robot.ik.add_task(reg_task)

    # Add some DOFTasks for a nicer posture
    robot.ik.add_task(DOFTask(
        robot, robot.R_SHOULDER_R, -0.5, gain=0.5, weight=1e-5))
    robot.ik.add_task(DOFTask(
        robot, robot.L_SHOULDER_R, +0.5, gain=0.5, weight=1e-5))

    # First, generate an initial posture
    robot.ik.verbosity = 2
    robot.ik.solve(max_it=100, impr_stop=1e-4)
    robot.ik.verbosity = 0

    # Next, we move the COM back and forth for 10 seconds
    move_com_back_and_forth(10)

    # Finally, we start the simulation with the IK on
    sim.schedule(robot.ik)
    sim.start()

    # Don't forget to give the user a prompt
    if IPython.get_ipython() is None:
        IPython.embed()
