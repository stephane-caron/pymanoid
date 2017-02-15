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

import openravepy
import time
import traceback

from numpy import array
from os import popen, system
from re import search
from threading import Thread

from misc import TimeStats


env = None  # global OpenRAVE environment
gravity = array([0, 0, -9.80665])  # gravity in world frame (ISO 80000-3)


def get_openrave_env():
    return env


class Process(object):

    """
    Processes implement the ``on_tick`` method called by the simulation.
    """

    def __init__(self):
        self._log_comp_times = False
        self.paused = False

    def log_comp_times(self, active=True):
        """
        Log average computation times for each tick.

        Parameters
        ----------
        active : bool, default=True
            Enable or disable logging.
        """
        self._log_comp_times = active

    def on_tick(self, sim):
        """
        Function called by the simulation at each clock tick.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.

        Note
        ----
        This function needs to be implemented by child classes.
        """
        raise NotImplementedError

    def pause(self):
        """Stop calling the process at new clock ticks."""
        self.paused = True

    def resume(self):
        """Resume calling the process at new clock ticks."""
        self.paused = False


class Simulation(object):

    """
    Simulation objects are the entry point of pymanoid scripts (similar to
    OpenRAVE environments, which they wrap).

    Parameters
    ----------
    dt : real
        Time interval between two ticks in simulation time.
    env_path : string, optional
        Load environment from XML/DAE file.
    env_xml : string, optional
        Load environment from XML string.
    """

    BACKGROUND_COLOR = [0.7, 0.8, 0.9]  # rien de tel que le bleu canard

    def __init__(self, dt, env_path=None, env_xml=None):
        global env, gravity
        if env is not None:
            raise Exception("an OpenRAVE environment already exists")
        env = openravepy.Environment()
        if env_path:
            env.Load(env_path)
        elif env_xml:
            env.LoadData(env_xml)
        env.GetPhysicsEngine().SetGravity(gravity)
        self.bodies = []
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

    @property
    def time(self):
        return self.tick_time * self.dt

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
            self._tick_processes()
            rem_time = self.dt - (time.time() - t0)
            if rem_time < 0.:
                print "sim.step(%d): warning: cycle time budget" % n,
                print "(%.1f ms) depleted!" % (self.dt * 1000.)
            self._tick_extras()
            rem_time = self.dt - (time.time() - t0)
            if rem_time > 1e-4:
                time.sleep(rem_time)
            self.tick_time += 1

    def _tick_processes(self):
        for process in self.processes:
            if not hasattr(process, 'paused'):
                most_likely_explanation = \
                    "did '%s' " % type(process).__name__ + \
                    "forget to call its parent constructor?"
                raise AttributeError(most_likely_explanation)
            if not process.paused:
                if process._log_comp_times:
                    t0i = time.time()
                    process.on_tick(self)
                    pname = type(process).__name__
                    self.log_comp_time(pname, time.time() - t0i)
                else:  # just do the thing
                    process.on_tick(self)

    def _tick_extras(self):
        if not self.extras:
            return
        for process in self.extras:
            try:
                if not process.paused:
                    process.on_tick(self)
            except Exception:
                msg = "SIMULATION: exception raised by extra process '%s'" % \
                    type(process).__name__
                print "\n%s\n%s" % ("-" * len(msg), msg)
                traceback.print_exc()
                print "%s\n" % ("-" * len(msg))

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
        Open OpenRAVE viewer.

        Parameters
        ----------
        plugin : string, optional
            Viewer plugin name ('qtcoin' or 'qtosg'), defaults to 'qtcoin'.
        """
        if self.viewer is not None:
            raise Exception("viewer is already set")
        self.env.SetViewer(plugin)
        self.viewer = self.env.GetViewer()
        self.viewer.SetBkgndColor(self.BACKGROUND_COLOR)
        self.set_camera_back(x=-3, y=0, z=0.7)

    def set_camera_back(self, x=None, y=None, z=None):
        x = self.viewer.GetCameraTransform()[0, 3] if x is None else x
        y = self.viewer.GetCameraTransform()[1, 3] if y is None else y
        z = self.viewer.GetCameraTransform()[2, 3] if z is None else z
        self.viewer.SetCamera([
            [0,  0, 1, x],
            [-1, 0, 0, y],
            [0, -1, 0, z],
            [0,  0, 0, 1.]])

    def set_camera_bottom(self, x=None, y=None, z=None):
        x = self.viewer.GetCameraTransform()[0, 3] if x is None else x
        y = self.viewer.GetCameraTransform()[1, 3] if y is None else y
        z = self.viewer.GetCameraTransform()[2, 3] if z is None else z
        self.viewer.SetCamera([
            [0, -1, 0, x],
            [1,  0, 0, y],
            [0,  0, 1, z],
            [0,  0, 0, 1]])

    def set_camera_front(self, x=None, y=None, z=None):
        x = self.viewer.GetCameraTransform()[0, 3] if x is None else x
        y = self.viewer.GetCameraTransform()[1, 3] if y is None else y
        z = self.viewer.GetCameraTransform()[2, 3] if z is None else z
        self.viewer.SetCamera([
            [0,  0, -1, x],
            [1,  0,  0, y],
            [0, -1,  0, z],
            [0,  0,  0, 1.]])

    def set_camera_left(self, x=None, y=None, z=None):
        x = self.viewer.GetCameraTransform()[0, 3] if x is None else x
        y = self.viewer.GetCameraTransform()[1, 3] if y is None else y
        z = self.viewer.GetCameraTransform()[2, 3] if z is None else z
        self.viewer.SetCamera([
            [-1, 0,  0, x],
            [0,  0, -1, y],
            [0, -1,  0, z],
            [0,  0,  0, 1.]])

    def set_camera_right(self, x=None, y=None, z=None):
        x = self.viewer.GetCameraTransform()[0, 3] if x is None else x
        y = self.viewer.GetCameraTransform()[1, 3] if y is None else y
        z = self.viewer.GetCameraTransform()[2, 3] if z is None else z
        self.viewer.SetCamera([
            [1,  0,  0, x],
            [0,  0, 1, y],
            [0, -1, 0, z],
            [0,  0, 0, 1.]])

    def set_camera_top(self, x=None, y=None, z=None):
        x = self.viewer.GetCameraTransform()[0, 3] if x is None else x
        y = self.viewer.GetCameraTransform()[1, 3] if y is None else y
        z = self.viewer.GetCameraTransform()[2, 3] if z is None else z
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
        self.window_id = "0x%s" % search('0x([0-9a-f]+)', line).group(1)

    def take_screenshot(self, fname):
        if self.window_id is None:
            self.read_window_id()
        system('import -window %s %s' % (self.window_id, fname))

    """
    Miscellaneous
    =============
    """

    def load_mesh(self, path, *args, **kwargs):
        """
        Load a pymanoid.Body from a DAE or VRML model.

        INPUT:

        - ``path`` -- path to DAE or VRML model
        """
        from body import Body
        assert path.endswith('.dae') or path.endswith('.wrl')
        if not self.env.Load(path):
            raise Exception("failed to load %s" % path)
        rave_body = self.env.GetBodies()[-1]
        body = Body(rave_body, *args, **kwargs)
        self.bodies.append(body)
        return body

    def log_comp_time(self, pname, ctime):
        """
        Log computation time for a given process.

        INPUT:

        - ``pname`` -- Process name
        - ``ctime`` -- computation time in [s] to log
        """
        if pname not in self.comp_times:
            self.comp_times[pname] = TimeStats()
        self.comp_times[pname].add(ctime)

    def print_comp_times(self, unit='ms'):
        total_avg, total_std = 0., 0.
        scale = {'s': 1, 'ms': 1000, 'us': 1e6}[unit]
        for (key, times) in self.comp_times.iteritems():
            print "%20d: %s" % (key, times.as_comp_times(unit))
            total_avg += scale * times.avg  # legit
            total_std += scale * times.std  # worst-case assumption
        print "%20s  ----------------------------------" % ''
        print "%20s: %.2f ms +/- %.2f ms" % ("total", total_avg, total_std)


class CameraRecorder(Process):

    def __init__(self, sim, output_folder, fname='video.mp4'):
        super(CameraRecorder, self).__init__()
        while output_folder.endswith('/'):
            output_folder = output_folder[:-1]
        sim.read_window_id()
        with open('%s/make_video.sh' % output_folder, 'w') as script:
            frate = 1. / sim.dt
            avconv = "avconv -r %f -qscale 1 -i %%05d.png %s" % (frate, fname)
            script.write("#!/bin/sh\n%s" % avconv)
        self.frame_index = 0
        self.output_folder = output_folder

    def on_tick(self, sim):
        fname = '%s/%05d.png' % (self.output_folder, self.frame_index)
        sim.take_screenshot(fname)
        self.frame_index += 1
