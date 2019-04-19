#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@lirmm.fr>
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
# You should have received a copy of the GNU Lesser General Public License
# along with pymanoid. If not, see <http://www.gnu.org/licenses/>.

import IPython
import numpy
import random

import pymanoid

from pymanoid import Contact
from pymanoid import PointMass
from pymanoid import Stance
from pymanoid.robots import JVRC1


def setup_ik_from_stance():
    """
    Setup inverse kinematics from the simpler stance interface.

    Notes
    -----
    This function is equivalent to :func:`setup_ik_from_tasks` below.
    """
    robot.set_z(0.8)  # hack to start with the robot above contacts
    com_target = robot.get_com_point_mass()
    lf_target = robot.left_foot.get_contact(pos=[0, 0.3, 0])
    rf_target = robot.right_foot.get_contact(pos=[0, -0.3, 0])
    stance = Stance(com=com_target, left_foot=lf_target, right_foot=rf_target)
    stance.dof_tasks[robot.R_SHOULDER_R] = -0.5
    stance.dof_tasks[robot.L_SHOULDER_R] = +0.5
    stance.bind(robot)


def setup_ik_from_tasks():
    """
    Setup the inverse kinematics task by task.

    Note
    -----
    This function is equivalent to :func:`setup_ik_from_stance` above.
    Beginners should take a look at that one first.

    Notes
    -----
    See <https://scaron.info/teaching/inverse-kinematics.html> for details.
    """
    from pymanoid.tasks import COMTask, ContactTask, DOFTask, PostureTask

    # Prepare targets
    lf_target = Contact(robot.sole_shape, pos=[0, 0.3, 0])
    rf_target = Contact(robot.sole_shape, pos=[0, -0.3, 0])

    # Initial robot pose
    robot.set_dof_values([0.8], dof_indices=[robot.TRANS_Z])
    com = PointMass(pos=robot.com, mass=robot.mass)

    # Prepare tasks
    left_foot_task = ContactTask(
        robot, robot.left_foot, lf_target, weight=1., gain=0.85)
    right_foot_task = ContactTask(
        robot, robot.right_foot, rf_target, weight=1., gain=0.85)
    com_task = COMTask(robot, com, weight=1e-2, gain=0.85)
    posture_task = PostureTask(robot, robot.q_halfsit, weight=1e-6, gain=0.85)

    # Add tasks to the IK solver
    robot.ik.add(left_foot_task)
    robot.ik.add(right_foot_task)
    robot.ik.add(com_task)
    robot.ik.add(posture_task)

    # Add shoulder DOF tasks for a nicer posture
    robot.ik.add(DOFTask(
        robot, robot.R_SHOULDER_R, -0.5, gain=0.5, weight=1e-5))
    robot.ik.add(DOFTask(
        robot, robot.L_SHOULDER_R, +0.5, gain=0.5, weight=1e-5))


if __name__ == '__main__':
    sim = pymanoid.Simulation(dt=0.03)
    robot = JVRC1('JVRC-1.dae', download_if_needed=True)
    robot.set_transparency(0.3)
    sim.set_viewer()
    sim.viewer.SetCamera([
        [-0.28985317,  0.40434422, -0.86746233,  2.73872042],
        [0.95680251,  0.10095043, -0.2726499,  0.86080128],
        [-0.02267371, -0.90901857, -0.41613837,  2.06654644],
        [0.,  0.,  0.,  1.]])

    # Optional: add low DOF acceleration limits
    robot.qdd_lim = 100. * numpy.ones(robot.q.shape)

    # Setup IK tasks
    if random.randint(0, 1) == 0:
        setup_ik_from_stance()
    else:  # the two functions are equivalent, pick one at random
        setup_ik_from_tasks()

    # Generate an initial posture by solving the IK problem
    robot.ik.verbosity = 2
    robot.ik.solve(max_it=100, impr_stop=1e-4)
    robot.ik.verbosity = 0

    # Next, track targets while the simulation runs
    sim.schedule(robot.ik)
    sim.start()

    print("""
===============================================================================

Robot posture generated.

From there, you can:

- Press [ESC] in the GUI to switch to edit mode.
- Click on the red box and drag-and-drop it around. The robot should follow.
- Click on the white background to de-select the box.
- Press [ESC] again to move the camera around.
- Click on a red contact slab under a robot foot and move the contact around.

===============================================================================
""")

    if IPython.get_ipython() is None:  # give the user a prompt
        IPython.embed()
