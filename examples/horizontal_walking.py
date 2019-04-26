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
from pymanoid.swing_foot import SwingFoot
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
    foot_spread : scalar
        Lateral distance between left and right foot centers.
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


class WalkingFSM(pymanoid.Process):

    """
    Finite State Machine for biped walking.

    Parameters
    ----------
    ssp_duration : scalar
        Duration of single-support phases, in [s].
    dsp_duration : scalar
        Duration of double-support phases, in [s].
    """

    def __init__(self, ssp_duration, dsp_duration):
        super(WalkingFSM, self).__init__()
        self.dsp_duration = dsp_duration
        self.next_footstep = 2
        self.ssp_duration = ssp_duration
        self.state = None
        #
        self.start_standing()

    def on_tick(self, sim):
        """
        Update function run at every simulation tick.

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
        """
        if self.state == "Standing":
            return self.run_standing()
        elif self.state == "DoubleSupport":
            return self.run_double_support()
        elif self.state == "SingleSupport":
            return self.run_single_support()
        raise Exception("Unknown state: " + self.state)

    def start_standing(self):
        """
        Switch to standing state.
        """
        self.start_walking = False
        self.state = "Standing"
        return self.run_standing()

    def run_standing(self):
        """
        Run standing state.
        """
        if self.start_walking:
            self.start_walking = False
            if self.next_footstep < len(footsteps):
                return self.start_double_support()

    def start_double_support(self):
        """
        Switch to double-support state.
        """
        self.rem_time = self.dsp_duration
        self.state = "DoubleSupport"
        return self.run_double_support()

    def run_double_support(self):
        """
        Run double-support state.
        """
        if self.rem_time <= 0.:
            return self.start_single_support()
        self.rem_time -= dt

    def start_single_support(self):
        """
        Switch to single-support state.
        """
        if self.next_footstep % 2 == 1:  # left foot swings
            self.stance_foot = stance.right_foot
            self.swing_foot = stance.left_foot
        else:  # right foot swings
            self.stance_foot = stance.left_foot
            self.swing_foot = stance.right_foot
        ssp_duration = self.ssp_duration
        swing_start = self.swing_foot
        swing_target = footsteps[self.next_footstep]
        self.next_footstep += 1
        self.rem_time = ssp_duration
        self.state = "SingleSupport"
        self.swing_interp = SwingFoot(
            swing_start, swing_target, ssp_duration, takeoff_clearance=0.075,
            landing_clearance=0.05)
        return self.run_single_support()

    def run_single_support(self):
        """
        Run single-support state.
        """
        if self.rem_time <= 0.:
            if self.next_footstep < len(footsteps):
                return self.start_double_support()
            else:  # footstep sequence is over
                return self.start_standing()
        self.swing_foot.set_pose(self.swing_interp.integrate(dt))
        # CoM: dummy code, to be replaced by linear model predictive control
        stance.com.set_x(0.5 * (self.swing_foot.x + self.stance_foot.x))
        self.rem_time -= dt


if __name__ == "__main__":
    numpy.random.seed(42)
    dt = 0.03  # [s]
    sim = pymanoid.Simulation(dt=dt)
    robot = JVRC1(download_if_needed=True)
    sim.set_viewer()
    sim.set_camera_transform([
        [-0.86825231, 0.13816899, -0.47649476, 2.95342016],
        [0.49606811, 0.22750768, -0.8379479, 3.26736617],
        [-0.0073722, -0.96392406, -0.2660753, 1.83063173],
        [0.,  0.,  0.,  1.]])
    robot.set_transparency(0.3)

    footsteps = generate_footsteps(
        distance=2.1,
        step_length=0.3,
        foot_spread=0.1,
        friction=0.7)
    com_target = PointMass([0, 0, robot.leg_length], robot.mass)
    stance = Stance(
        com=com_target,
        left_foot=footsteps[0].copy(hide=True),
        right_foot=footsteps[1].copy(hide=True))
    stance.bind(robot)
    fsm = WalkingFSM(ssp_duration=0.7, dsp_duration=0.1)

    # robot.ik.DEFAULT_WEIGHTS['POSTURE'] = 1e-5
    robot.ik.solve(max_it=42)
    robot.ik.add(DOFTask(robot, robot.WAIST_P, 0.2, weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.WAIST_Y, 0., weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.WAIST_R, 0., weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.ROT_P, 0., weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.R_SHOULDER_R, -0.5, weight=1e-3))
    robot.ik.add(DOFTask(robot, robot.L_SHOULDER_R, 0.5, weight=1e-3))
    robot.ik.add(MinCAMTask(robot, weight=1e-4))
    robot.ik.solve(max_it=24)

    sim.schedule(fsm)
    sim.schedule(robot.ik, log_comp_times=True)

    com_traj_drawer = TrajectoryDrawer(robot.stance.com, 'b-')
    lf_traj_drawer = TrajectoryDrawer(robot.left_foot, 'g-')
    # preview_drawer = PreviewDrawer()
    rf_traj_drawer = TrajectoryDrawer(robot.right_foot, 'r-')
    # wrench_drawer = PointMassWrenchDrawer(com_target, lambda: fsm.cur_stance)

    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(lf_traj_drawer)
    # sim.schedule_extra(preview_drawer)
    sim.schedule_extra(rf_traj_drawer)
    # sim.schedule_extra(wrench_drawer)

    sim.start()

    def start_walking():
        fsm.start_walking = True

    print("""

Linear Model Predictive Control
===============================

Ready to go! Start walking by running:

    start_walking()

""")

    start_walking()
    if IPython.get_ipython() is None:
        IPython.embed()
