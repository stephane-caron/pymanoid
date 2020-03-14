#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2020 Stephane Caron <stephane.caron@normalesup.org>
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
This example implements a walking pattern generator for horizontal walking
based on linear model predictive control
<https://hal.inria.fr/file/index/docid/390462/filename/Preview.pdf>.
"""

import IPython

from numpy import array

import pymanoid

from pymanoid.body import PointMass
from pymanoid.contact import Contact
from pymanoid.gui import RobotWrenchDrawer
from pymanoid.gui import TrajectoryDrawer
from pymanoid.mpc import LinearPredictiveControl
from pymanoid.robots import JVRC1
from pymanoid.stance import Stance
from pymanoid.swing_foot import SwingFoot


def generate_footsteps(distance, step_length, foot_spread, friction):
    """
    Generate a set of footsteps for walking forward.

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
    append_contact(x, -y)  # here x == distance
    return contacts


class HorizontalWalkingFSM(pymanoid.Process):

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
        super(HorizontalWalkingFSM, self).__init__()
        self.dsp_duration = dsp_duration
        self.mpc_timestep = round(0.1 / dt) * dt  # update MPC every ~0.1 [s]
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
        if self.next_footstep % 2 == 1:  # left foot swings
            self.stance_foot = stance.right_foot
        else:  # right foot swings
            self.stance_foot = stance.left_foot
        dsp_duration = self.dsp_duration
        if self.next_footstep == 2 or self.next_footstep == len(footsteps) - 1:
            # double support is a bit longer for the first and last steps
            dsp_duration = 4 * self.dsp_duration
        self.swing_target = footsteps[self.next_footstep]
        self.rem_time = dsp_duration
        self.state = "DoubleSupport"
        self.start_com_mpc_dsp()
        return self.run_double_support()

    def start_com_mpc_dsp(self):
        self.update_mpc(self.rem_time, self.ssp_duration)

    def run_double_support(self):
        """
        Run double-support state.
        """
        if self.rem_time <= 0.:
            return self.start_single_support()
        self.run_com_mpc()
        self.rem_time -= dt

    def start_single_support(self):
        """
        Switch to single-support state.
        """
        if self.next_footstep % 2 == 1:  # left foot swings
            self.swing_foot = stance.free_contact('left_foot')
        else:  # right foot swings
            self.swing_foot = stance.free_contact('right_foot')
        self.next_footstep += 1
        self.rem_time = self.ssp_duration
        self.state = "SingleSupport"
        self.start_swing_foot()
        self.start_com_mpc_ssp()
        return self.run_single_support()

    def start_swing_foot(self):
        """
        Initialize swing foot interpolator for single-support state.
        """
        self.swing_start = self.swing_foot.pose
        self.swing_interp = SwingFoot(
            self.swing_foot, self.swing_target, self.ssp_duration,
            takeoff_clearance=0.05, landing_clearance=0.05)

    def start_com_mpc_ssp(self):
        self.update_mpc(0., self.rem_time)

    def run_single_support(self):
        """
        Run single-support state.
        """
        if self.rem_time <= 0.:
            stance.set_contact(self.swing_foot)
            if self.next_footstep < len(footsteps):
                return self.start_double_support()
            else:  # footstep sequence is over
                return self.start_standing()
        self.run_swing_foot()
        self.run_com_mpc()
        self.rem_time -= dt

    def run_swing_foot(self):
        """
        Run swing foot interpolator for single-support state.
        """
        self.swing_foot.set_pose(self.swing_interp.integrate(dt))

    def update_mpc(self, dsp_duration, ssp_duration):
        nb_preview_steps = 16
        T = self.mpc_timestep
        nb_init_dsp_steps = int(round(dsp_duration / T))
        nb_init_ssp_steps = int(round(ssp_duration / T))
        nb_dsp_steps = int(round(self.dsp_duration / T))
        A = array([[1., T, T ** 2 / 2.], [0., 1., T], [0., 0., 1.]])
        B = array([T ** 3 / 6., T ** 2 / 2., T]).reshape((3, 1))
        h = stance.com.z
        g = -sim.gravity[2]
        zmp_from_state = array([1., 0., -h / g])
        C = array([+zmp_from_state, -zmp_from_state])
        D = None
        e = [[], []]
        cur_vertices = self.stance_foot.get_scaled_contact_area(0.9)
        next_vertices = self.swing_target.get_scaled_contact_area(0.9)
        for coord in [0, 1]:
            cur_max = max(v[coord] for v in cur_vertices)
            cur_min = min(v[coord] for v in cur_vertices)
            next_max = max(v[coord] for v in next_vertices)
            next_min = min(v[coord] for v in next_vertices)
            e[coord] = [
                array([+1000., +1000.]) if i < nb_init_dsp_steps
                else array([+cur_max, -cur_min])
                if i - nb_init_dsp_steps <= nb_init_ssp_steps
                else array([+1000., +1000.])
                if i - nb_init_dsp_steps - nb_init_ssp_steps < nb_dsp_steps
                else array([+next_max, -next_min])
                for i in range(nb_preview_steps)]
        self.x_mpc = LinearPredictiveControl(
            A, B, C, D, e[0],
            x_init=array([stance.com.x, stance.com.xd, stance.com.xdd]),
            x_goal=array([self.swing_target.x, 0., 0.]),
            nb_steps=nb_preview_steps,
            wxt=1., wu=0.01)
        self.y_mpc = LinearPredictiveControl(
            A, B, C, D, e[1],
            x_init=array([stance.com.y, stance.com.yd, stance.com.ydd]),
            x_goal=array([self.swing_target.y, 0., 0.]),
            nb_steps=nb_preview_steps,
            wxt=1., wu=0.01)
        self.x_mpc.solve()
        self.y_mpc.solve()
        self.preview_time = 0.

    def plot_mpc_preview(self):
        import pylab
        T = self.mpc_timestep
        h = stance.com.z
        g = -sim.gravity[2]
        trange = [sim.time + k * T for k in range(len(self.x_mpc.X))]
        pylab.ion()
        pylab.clf()
        pylab.subplot(211)
        pylab.plot(trange, [v[0] for v in self.x_mpc.X])
        pylab.plot(trange, [v[0] - v[2] * h / g for v in self.x_mpc.X])
        pylab.subplot(212)
        pylab.plot(trange, [v[0] for v in self.y_mpc.X])
        pylab.plot(trange, [v[0] - v[2] * h / g for v in self.y_mpc.X])

    def run_com_mpc(self):
        """
        Run CoM predictive control for single-support state.
        """
        if self.preview_time >= self.mpc_timestep:
            if self.state == "DoubleSupport":
                self.update_mpc(self.rem_time, self.ssp_duration)
            else:  # self.state == "SingleSupport":
                self.update_mpc(0., self.rem_time)
        com_jerk = array([self.x_mpc.U[0][0], self.y_mpc.U[0][0], 0.])
        stance.com.integrate_constant_jerk(com_jerk, dt)
        self.preview_time += dt


if __name__ == "__main__":
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
    stance = Stance(
        com=PointMass([0, 0, robot.leg_length], robot.mass),
        left_foot=footsteps[0].copy(hide=True),
        right_foot=footsteps[1].copy(hide=True))
    stance.bind(robot)
    robot.ik.solve(max_it=42)

    ssp_duration = round(0.7 / dt) * dt  # close to 0.7 [s]
    dsp_duration = round(0.1 / dt) * dt  # close to 0.1 [s]
    fsm = HorizontalWalkingFSM(ssp_duration, dsp_duration)

    sim.schedule(fsm)
    sim.schedule(robot.ik, log_comp_times=True)
    sim.schedule(robot.wrench_distributor, log_comp_times=True)

    com_traj_drawer = TrajectoryDrawer(robot.stance.com, 'b-')
    lf_traj_drawer = TrajectoryDrawer(robot.left_foot, 'g-')
    rf_traj_drawer = TrajectoryDrawer(robot.right_foot, 'r-')
    wrench_drawer = RobotWrenchDrawer(robot)

    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(lf_traj_drawer)
    sim.schedule_extra(rf_traj_drawer)
    sim.schedule_extra(wrench_drawer)

    sim.start()

    def start_walking():
        fsm.start_walking = True

    start_walking()  # comment this out to start walking manually
    if IPython.get_ipython() is None:
        IPython.embed()
