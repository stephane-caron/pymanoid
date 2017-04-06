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
import numpy
import os
import sys
import time

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../')
    import pymanoid

from pymanoid import Contact, Cube
from pymanoid.robots import JVRC1
from pymanoid.tasks import COMTask, ContactTask, DOFTask, PostureTask


def move_com_back_and_forth(duration, dt=1e-2):
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

    # Initial robot pose
    robot.set_transparency(0.4)
    dof_targets = [  # will also be passed to the IK
        (robot.R_SHOULDER_R, -.5),
        (robot.L_SHOULDER_R, +.5)]
    q_init = robot.q.copy()
    for (dof_id, dof_ref) in dof_targets:
        robot.set_dof_values([dof_ref], [dof_id])
        q_init[dof_id] = dof_ref
    robot.set_dof_values([-1], [robot.R_SHOULDER_P])
    robot.set_dof_values([-1], [robot.L_SHOULDER_P])
    robot.set_dof_values([0.8], dof_indices=[robot.TRANS_Z])
    init_com = robot.com.copy()

    # IK targets
    com = Cube(0.05, pos=robot.com, color='g')
    lf_target = Contact(robot.sole_shape, pos=[0, 0.3, 0], visible=True)
    rf_target = Contact(robot.sole_shape, pos=[0, -0.3, 0], visible=True)

    # IK tasks
    lf_task = ContactTask(robot, robot.left_foot, lf_target, weight=1000)
    rf_task = ContactTask(robot, robot.right_foot, rf_target, weight=1000)
    com_task = COMTask(robot, com, weight=10)
    reg_task = PostureTask(robot, robot.q, weight=0.1)  # regularization task

    # IK setup
    robot.init_ik(active_dofs=robot.whole_body)
    robot.ik.add_task(lf_task)
    robot.ik.add_task(rf_task)
    robot.ik.add_task(com_task)
    robot.ik.add_task(reg_task)
    for (dof_id, dof_ref) in dof_targets:
        robot.ik.add_task(
            DOFTask(robot, dof_id, dof_ref, gain=0.5, weight=0.1))

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
