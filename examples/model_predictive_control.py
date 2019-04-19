#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Stephane Caron <stephane.caron@lirmm.fr>
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

"""
This example implements a walking pattern generator for horizontal walking
based on linear model predictive control
<https://hal.inria.fr/file/index/docid/390462/filename/Preview.pdf>.
"""

import IPython
import numpy

import pymanoid

from pymanoid.body import PointMass
from pymanoid.contact import Contact
from pymanoid.gui import TrajectoryDrawer
from pymanoid.robots import JVRC1
from pymanoid.stance import Stance
from pymanoid.tasks import DOFTask
from pymanoid.tasks import MinCAMTask


def generate_footsteps(distance, step_length, foot_spread, friction):
    """
    Generate a new slanted staircase with tilted steps.

    Parameters
    ----------
    distance : scalar
        Total distance to walk forward in [m].
    step_length : scalar
        Distance between right and left heel in double support.
    friction : scalar
        Friction coefficient between a robot foot and a step.
    """
    contacts = []

    def append_contact(x, y):
        contacts.append(Contact(
            shape=robot.sole_shape, pos=[x, y, 0.], friction=friction))

    append_contact(0., +foot_spread)
    append_contact(0., -foot_spread)
    x = 0.
    y = foot_spread
    while x < distance:
        if distance - x <= step_length:
            x += min(distance - x, 0.5 * step_length)
        else:  # still way to go
            x += step_length
        y *= -1.
        append_contact(x, y)
    # here x == distance
    y *= -1.
    append_contact(x, y)
    return contacts


if __name__ == "__main__":
    numpy.random.seed(42)
    sim = pymanoid.Simulation(dt=0.03)
    robot = JVRC1(download_if_needed=True)
    sim.set_viewer()
    sim.set_camera_transform([
        [-0.86825231, 0.13816899, -0.47649476, 2.95342016],
        [0.49606811, 0.22750768, -0.8379479, 3.26736617],
        [-0.0073722, -0.96392406, -0.2660753, 1.83063173],
        [0.,  0.,  0.,  1.]])
    robot.set_transparency(0.3)

    com_target = PointMass([0, 0, 0.85], robot.mass)
    footsteps = generate_footsteps(
        distance=2.1,
        step_length=0.3,
        foot_spread=0.1,
        friction=0.7)

    stance = Stance(
        com=com_target,
        left_foot=footsteps[0],
        right_foot=footsteps[1])
    stance.bind(robot)

    # robot.ik.DEFAULT_WEIGHTS['POSTURE'] = 1e-5
    robot.ik.solve(max_it=50)
    robot.ik.add(DOFTask(robot, robot.WAIST_P, 0.2, weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.WAIST_Y, 0., weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.WAIST_R, 0., weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.ROT_P, 0., weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.R_SHOULDER_R, -0.5, weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.L_SHOULDER_R, 0.5, weight=1e-3))
    robot.ik.add(MinCAMTask(robot, weight=1e-4))

    sim.schedule(robot.ik, log_comp_times=True)

    com_traj_drawer = TrajectoryDrawer(com_target, 'b-')
    lf_traj_drawer = TrajectoryDrawer(robot.left_foot, 'g-')
    # preview_drawer = PreviewDrawer()
    rf_traj_drawer = TrajectoryDrawer(robot.right_foot, 'r-')
    # wrench_drawer = PointMassWrenchDrawer(com_target, lambda: fsm.cur_stance)

    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(lf_traj_drawer)
    # sim.schedule_extra(preview_drawer)
    sim.schedule_extra(rf_traj_drawer)
    # sim.schedule_extra(wrench_drawer)

    print("""

Linear Model Predictive Control
===============================

Ready to go! You can control the simulation by:

    sim.start() -- run/resume simulation in a separate thread
    sim.step(100) -- run simulation in current thread for 100 steps
    sim.stop() -- stop/pause simulation

You can access all state variables via this IPython shell.
Here is the list of global objects. Use <TAB> to see what's inside.

    robot -- kinematic model of the robot (includes IK solver)

Enjoy :)

""")

    if IPython.get_ipython() is None:
        IPython.embed()
