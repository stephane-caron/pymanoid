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

import openravepy
import time

from numpy import array
from os import popen, system
from re import search
from threading import Thread

from misc import AvgStdEstimator


env = None  # global OpenRAVE environment


def get_openrave_env():
    return env


class Simulation(object):

    BACKGROUND_COLOR = [1., 1., 1.]

    def __init__(self, dt=3e-2, env_path=None, env_xml=None):
        """
        Create a new simulation object and initialize its OpenRAVE environment.

        INPUT:

        - ``dt`` -- time interval between two ticks in simulation time
        - ``env_path`` -- (optional) load environment from XML/DAE file
        - ``env_xml`` -- (optional) load environment from XML string
        """
        global env
        if env is not None:
            raise Exception("an OpenRAVE environment already exists")
        env = openravepy.Environment()
        gravity = array([0, 0, -9.80665])  # ISO 80000-3
        if env_path:
            env.Load(env_path)
        elif env_xml:
            env.LoadData(env_xml)
        env.GetPhysicsEngine().SetGravity(gravity)
        self.comp_times = {}
        self.dt = dt
        self.env = env
        self.extras = []
        self.gravity = gravity
        self.is_running = False
        self.processes = []
        self.tick_time = 0
        self.viewer = None
        self.window_id = None

    def __del__(self):
        """Close thread at shutdown."""
        self.stop()

    def schedule(self, process):
        """Add a Process to the schedule list (insertion order matters)."""
        self.processes.append(process)

    def schedule_extra(self, process):
        """Schedule a Process not counted in the computation time budget."""
        self.extras.append(process)

    def step(self, n=1):
        """Perform one simulation step."""
        for _ in xrange(n):
            t0 = time.time()
            for process in self.processes:
                process.on_tick(self)
            rem_time = self.dt - (time.time() - t0)
            # TODO: internal sim/real time estimate
            if self.extras:
                for process in self.extras:
                    process.on_tick(self)
                rem_time = self.dt - (time.time() - t0)
            if rem_time > 1e-4:
                time.sleep(rem_time)
            self.tick_time += 1

    """
    Threading
    =========
    """

    def run_thread(self):
        """Run simulation thread."""
        while self.is_running:
            self.step()

    def start(self):
        """Start simulation thread. """
        self.is_running = True
        self.thread = Thread(target=self.run_thread, args=())
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        self.is_running = False

    """
    Viewer
    ======
    """

    def set_viewer(self, plugin='qtcoin'):
        """
        Set OpenRAVE viewer.

        INPUT:

        - ``plugin`` -- (optional) viewer plugin name, e.g. 'qtcoin' or 'qtosg'
        """
        if self.viewer is not None:
            raise Exception("viewer is already set")
        self.env.SetViewer(plugin)
        self.viewer = self.env.GetViewer()
        self.viewer.SetBkgndColor(self.BACKGROUND_COLOR)
        self.set_camera_back()

    def set_camera_back(self, x=-3., y=0., z=0.7):
        self.viewer.SetCamera([
            [0,  0, 1, x],
            [-1, 0, 0, y],
            [0, -1, 0, z],
            [0,  0, 0, 1.]])

    def set_camera_bottom(self, x=0., y=0., z=-2.):
        self.viewer.SetCamera([
            [0, -1, 0, x],
            [1,  0, 0, y],
            [0,  0, 1, z],
            [0,  0, 0, 1]])

    def set_camera_front(self, x=3., y=0., z=0.7):
        self.viewer.SetCamera([
            [0,  0, -1, x],
            [1,  0,  0, y],
            [0, -1,  0, z],
            [0,  0,  0, 1.]])

    def set_camera_left(self, x=0., y=3., z=0.7):
        self.viewer.SetCamera([
            [-1, 0,  0, x],
            [0,  0, -1, y],
            [0, -1,  0, z],
            [0,  0,  0, 1.]])

    def set_camera_right(self, x=0., y=-3., z=0.7):
        self.viewer.SetCamera([
            [1,  0,  0, x],
            [0,  0, 1, y],
            [0, -1, 0, z],
            [0,  0, 0, 1.]])

    def set_camera_top(self, x=0., y=0., z=3.):
        self.viewer.SetCamera([
            [0, -1,  0, x],
            [-1, 0,  0, y],
            [0,  0, -1, z],
            [0,  0,  0, 1.]])

    """
    Screnshots
    ==========
    """

    def read_window_id(self):
        print "Please click on the OpenRAVE window."
        line = popen('/usr/bin/xwininfo | grep "Window id:"').readlines()[0]
        __window_id__ = "0x%s" % search('0x([0-9a-f]+)', line).group(1)
        print "Window id:", __window_id__

    def take_screenshot(self, fname):
        if self.window_id is None:
            self.read_window_id()
        system('import -window %s %s' % (self.window_id, fname))

    """
    Logging
    =======
    """

    def report_comp_times(self, d):
        for (key, value) in d.iteritems():
            if key not in self.comp_times:
                self.comp_times[key] = AvgStdEstimator()
            self.comp_times[key].add(value)

    def print_comp_times(self):
        total_avg, total_std = 0., 0.
        for (key, estimator) in self.comp_times.iteritems():
            avg, std, n = estimator.get_all()
            avg *= 1000  # [ms]
            std *= 1000  # [ms]
            print "%20s: %.1f ms +/- %.1f ms over %5d items" % (
                key, avg, std, n)
            total_avg += avg
            total_std += std
        print "%20s  ----------------------------------" % ''
        print "%20s: %.1f ms +/- %.1f ms" % ("total", total_avg, total_std)


class Process(object):

    """Processes implement the ``on_tick`` method called by the Simulation."""

    def on_tick(self, sim):
        """Function called by the Simulation parent after each clock tick."""
        raise NotImplementedError
