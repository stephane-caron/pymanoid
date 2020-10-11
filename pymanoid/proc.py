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

from numpy import zeros


class Process(object):

    """
    Processes implement the ``on_tick`` method called by the simulation.
    """

    def __init__(self):
        self.log_comp_times = False
        self.is_paused = False

    def on_tick(self, sim):
        """
        Main function called by the simulation at each control cycle.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        """
        raise NotImplementedError("should be implemented by child class")

    def pause(self):
        """
        Stop calling the process at new clock ticks.
        """
        self.is_paused = True

    def resume(self):
        """
        Resume calling the process at new clock ticks.
        """
        self.is_paused = False


class CameraRecorder(Process):

    """
    Video recording process.

    When created, this process will ask the user to click on the OpenRAVE GUI
    to get its window ID. Then, it will save a screenshot in the camera folder
    at each tick of the simulation (don't expect real-time recording...). When
    your simulation is over, go to the camera folder and run the script called
    ``make_video.sh``.

    Parameters
    ----------
    sim : Simulation
        Simulation instance.
    fname : string, optional
        Video file name.
    tmp_folder : string, optional
        Temporary folder where screenshots will be recorded.

    Note
    ----
    Creating videos requires the following dependencies (here listed for Ubuntu
    14.04): ``sudo apt-get install x11-utils imagemagick libav-tools``.

    Note
    ----
    Don't expect the simulation to run real-time while recording.

    Note
    ----
    The GUI window should stay visible on your screen for the whole duration of
    the recording. Also, don't resize it, otherwise video conversion will fail
    later on.
    """

    def __init__(self, sim, fname=None, tmp_folder='pymanoid_rec'):
        super(CameraRecorder, self).__init__()
        if fname is None:
            now = datetime.datetime.now()
            fname = now.strftime('pymanoid-%Y-%m-%d-%H%M%S.mp4')
        while tmp_folder.endswith('/'):
            tmp_folder = tmp_folder[:-1]
        sim.get_viewer_window_id()
        if not os.path.exists(tmp_folder):
            os.makedirs(tmp_folder)
        script_name = 'make_pymanoid_video.sh'
        with open(script_name, 'w') as script:
            frate = int(1. / sim.dt)
            script.write(
                ("#!/bin/sh\n") +
                (("avconv -r %d" % frate) +
                 (" -i %s/%%05d.png" % tmp_folder) +
                 (" -vf crop=\"trunc(iw/2)*2:trunc(ih/2)*2:0:0\"") +
                 (" %s" % fname)) +
                (" && rm -rf %s" % tmp_folder) +
                (" && rm %s" % script_name))
        st = fstat(script_name)
        chmod(script_name, st.st_mode | S_IEXEC)
        self.frame_index = 0
        self.sim = sim
        self.tmp_folder = tmp_folder

    def on_tick(self, sim):
        """
        Main function called by the simulation at each control cycle.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        """
        fname = '%s/%05d.png' % (self.tmp_folder, self.frame_index)
        sim.take_screenshot(fname)
        self.frame_index += 1

    def wait_for(self, wait_time):
        """
        Pause the video by repeating the last frame for a certain duration.

        Parameters
        ----------
        wait_time : scalar
            Duration in [s].
        """
        for _ in range(int(wait_time / self.sim.dt)):
            self.on_tick(self.sim)


class JointRecorder(Process):

    """
    Record joint trajectories (position, velocity, acceleration, torque) from
    the robot model.

    Parameters
    ----------
    robot : Robot
        Target robot state to record from.
    """

    def __init__(self, robot):
        super(JointRecorder, self).__init__()
        self.q = [robot.q]
        self.qd = [robot.qd]
        self.qdd = [zeros(robot.q.shape)]
        self.robot = robot
        self.tau = [zeros(robot.q.shape)]
        self.times = [0.]

    def on_tick(self, sim):
        """
        Main function called by the simulation at each control cycle.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        """
        qd_prev = self.qd[-1]
        qdd = (self.robot.qd - qd_prev) / sim.dt
        tm, tc, tg = self.robot.compute_inverse_dynamics(qdd)
        self.q.append(self.robot.q)
        self.qd.append(self.robot.qd)
        self.qdd.append(qdd)
        self.tau.append(tm + tc + tg)
        self.times.append(sim.time)

    def plot(self, dofs=None):
        """
        Plot recorded joint trajectories.

        Parameters
        ----------
        dofs : list of integers
            DOF indices to plot trajectories for, e.g. ``robot.left_leg``.
        """
        import pylab
        if dofs is None:
            dofs = range(self.robot.nb_dofs)
        plots = [
            (self.q, "Joint angles [rad]"),
            (self.qd, "Joint velocities [rad/s]"),
            (self.qdd, "Joint accelerations [rad/s^2]"),
            (self.tau, "Joint torques [Nm]")]
        for i, plot in enumerate(plots):
            traj, ylabel = plot
            pylab.subplot(411 + i)
            pylab.plot(self.times, [v[dofs] for v in traj])
            pylab.grid(True)
            pylab.xlabel('Time [s]')
            pylab.ylabel(ylabel)
            pylab.xlim(0., self.times[-1])
