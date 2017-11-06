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

from numpy import arange, array
from os import chmod, popen, system
from os import stat as fstat
from re import search
from stat import S_IEXEC
from threading import Lock, Thread
from time import sleep, time

from misc import Statistics


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
        self.is_paused = False

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
        """
        raise NotImplementedError("should be implemented by child class")

    def pause(self):
        """Stop calling the process at new clock ticks."""
        self.is_paused = True

    def resume(self):
        """Resume calling the process at new clock ticks."""
        self.is_paused = False


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

    BACKGROUND_COLOR = [0.7, 0.8, 0.9]

    # BACKGROUND_COLOR = [0.7, 0.80, 0.9]  # color for 0.6.* series
    # BACKGROUND_COLOR = [1.0, 0.98, 0.9]  # color for 0.7.* series

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
        self.lock = None
        self.processes = []
        self.nb_steps = 0
        self.slowdown = 1.
        self.viewer = None
        self.watch_comp_times = False
        self.window_id = None

    @property
    def camera_transform(self):
        """Camera transform in the world frame."""
        return self.viewer.GetCameraTransform()

    @property
    def time(self):
        return self.nb_steps * self.dt

    def __del__(self):
        self.stop()

    def schedule(self, process, paused=False):
        """
        Add a process to the schedule list.

        Parameters
        ----------
        process : pymanoid.Process
            Process that will be called at every simulation tick.
        paused : bool, optional
            If True, the process will be initially paused.

        Note
        ----
        The order in which processes are scheduled does matter.
        """
        assert issubclass(type(process), Process), \
            "Cannot schedule process of type %s" % type(process).__name__
        self.processes.append(process)
        if paused:
            process.pause()

    def schedule_extra(self, process):
        """Schedule a Process not counted in the computation time budget."""
        self.extras.append(process)

    def step(self, n=1):
        """
        Perform a given number of simulation steps (default is one).

        Parameters
        ----------
        n : int, optional
            Number of simulation steps.
        """
        for _ in xrange(n):
            t0 = time()
            self._tick_processes()
            rem_time = self.dt - (time() - t0)
            if __debug__ and self.watch_comp_times and rem_time < 0.:
                print "Simulation warning: cycle time budget",
                print "(%.1f ms) depleted" % (self.dt * 1000.)
            self._tick_extras()
            rem_time = self.dt - (time() - t0)
            if rem_time > 1e-4:
                sleep(rem_time)
            if self.slowdown > 1.01:
                sleep((self.slowdown - 1.) * self.dt)
            self.nb_steps += 1

    def _tick_processes(self):
        for process in self.processes:
            if not hasattr(process, 'is_paused'):
                most_likely_explanation = \
                    "did '%s' " % type(process).__name__ + \
                    "forget to call its parent constructor?"
                raise AttributeError(most_likely_explanation)
            if not process.is_paused:
                if process._log_comp_times:
                    t0i = time()
                    process.on_tick(self)
                    pname = type(process).__name__
                    self.log_comp_time(pname, time() - t0i)
                else:  # just do the thing
                    process.on_tick(self)

    def _tick_extras(self):
        if not self.extras:
            return
        for process in self.extras:
            if not process.is_paused:
                process.on_tick(self)

    """
    Threading
    =========
    """

    def run_thread(self):
        """Run simulation thread."""
        while self.is_running:
            with self.lock:
                self.step()

    def start(self):
        """Start simulation thread. """
        self.is_running = True
        self.lock = Lock()
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
        if self.viewer is None:  # seldom happens that GetViewer() is not
            sleep(0.01)  # immediately available after SetViewer()
            self.viewer = self.env.GetViewer()
        self.viewer.SetBkgndColor(self.BACKGROUND_COLOR)
        self.set_camera_back(x=-3, y=0, z=0.7)

    def set_camera_back(self, x=None, y=None, z=None):
        x = self.camera_transform[0, 3] if x is None else x
        y = self.camera_transform[1, 3] if y is None else y
        z = self.camera_transform[2, 3] if z is None else z
        self.viewer.SetCamera([
            [0,  0, 1, x],
            [-1, 0, 0, y],
            [0, -1, 0, z],
            [0,  0, 0, 1.]])

    def set_camera_bottom(self, x=None, y=None, z=None):
        x = self.camera_transform[0, 3] if x is None else x
        y = self.camera_transform[1, 3] if y is None else y
        z = self.camera_transform[2, 3] if z is None else z
        self.viewer.SetCamera([
            [0, -1, 0, x],
            [1,  0, 0, y],
            [0,  0, 1, z],
            [0,  0, 0, 1]])

    def set_camera_front(self, x=None, y=None, z=None):
        x = self.camera_transform[0, 3] if x is None else x
        y = self.camera_transform[1, 3] if y is None else y
        z = self.camera_transform[2, 3] if z is None else z
        self.viewer.SetCamera([
            [0,  0, -1, x],
            [1,  0,  0, y],
            [0, -1,  0, z],
            [0,  0,  0, 1.]])

    def set_camera_left(self, x=None, y=None, z=None):
        x = self.camera_transform[0, 3] if x is None else x
        y = self.camera_transform[1, 3] if y is None else y
        z = self.camera_transform[2, 3] if z is None else z
        self.viewer.SetCamera([
            [-1, 0,  0, x],
            [0,  0, -1, y],
            [0, -1,  0, z],
            [0,  0,  0, 1.]])

    def set_camera_right(self, x=None, y=None, z=None):
        x = self.camera_transform[0, 3] if x is None else x
        y = self.camera_transform[1, 3] if y is None else y
        z = self.camera_transform[2, 3] if z is None else z
        self.viewer.SetCamera([
            [1,  0,  0, x],
            [0,  0, 1, y],
            [0, -1, 0, z],
            [0,  0, 0, 1.]])

    def set_camera_top(self, x=None, y=None, z=None):
        x = self.camera_transform[0, 3] if x is None else x
        y = self.camera_transform[1, 3] if y is None else y
        z = self.camera_transform[2, 3] if z is None else z
        self.viewer.SetCamera([
            [0, -1,  0, x],
            [-1, 0,  0, y],
            [0,  0, -1, z],
            [0,  0,  0, 1.]])

    def move_camera_to(self, T, duration=0., dt=3e-2):
        T_i = self.camera_transform
        for t in arange(0., duration, dt):
            self.viewer.SetCamera((1. - t) * T_i + t * T)
            sleep(dt)
        self.viewer.SetCamera(T)

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

    def load_mesh(self, path):
        """
        Load a pymanoid.Body from a DAE or VRML model.

        Parameters
        ----------
        path : string
            Path to DAE or VRML model.
        """
        from body import Body
        assert path.endswith('.dae') \
            or path.endswith('.wrl') or path.endswith('.xml')
        if not self.env.Load(path):
            raise Exception("failed to load %s" % path)
        rave_body = self.env.GetBodies()[-1]
        body = Body(rave_body)
        self.bodies.append(body)
        return body

    def log_comp_time(self, label, ctime):
        """
        Log computation time for a given process.

        Parameters
        ----------
        label : string
            Label of the operation.
        ctime : scalar
            Computation time in [s].
        """
        if label not in self.comp_times:
            self.comp_times[label] = Statistics()
        self.comp_times[label].add(ctime)

    def print_comp_times(self, unit='ms'):
        total_avg, total_std = 0., 0.
        scale = {'s': 1, 'ms': 1000, 'us': 1e6}[unit]
        for key in sorted(self.comp_times):
            times = self.comp_times[key]
            print "%20s: %s" % (key, times.as_comp_times(unit))
            total_avg += scale * times.avg  # legit
            total_std += scale * times.std  # worst-case assumption
        print "%20s  ----------------------------------" % ''
        print "%20s: %.2f ms +/- %.2f ms" % ("total", total_avg, total_std)


class CameraRecorder(Process):

    """
    Video recording process.

    Parameters
    ----------
    sim : Simulation
        Global simulation.
    output_folder : string
        Path where screenshots will be recorded.
    fname : string
        Video file name.

    Note
    ----
    Creating videos requires the following dependencies (here listed as package
    names for Ubuntu 14.04): ``x11-utils``, ``imagemagick``, ``libav-tools``.

    Notes
    -----
    When created, this process will ask the user to click on the OpenRAVE GUI to
    get its window ID. Then, it will save a screenshot in the camera folder at
    each tick of the simulation (don't expect real-time recording...). When your
    simulation is over, go to the camera folder and run the script called
    ``make_video.sh``.
    """
    def __init__(self, sim, output_folder, fname='video.mp4'):
        super(CameraRecorder, self).__init__()
        while output_folder.endswith('/'):
            output_folder = output_folder[:-1]
        sim.read_window_id()
        script_path = '%s/make_video.sh' % output_folder
        with open(script_path, 'w') as script:
            frate = int(1. / sim.dt)
            avconv = "avconv -r %d -qscale 1 -i %%05d.png %s" % (frate, fname)
            script.write("#!/bin/sh\n%s" % avconv)
        st = fstat(script_path)
        chmod(script_path, st.st_mode | S_IEXEC)
        self.frame_index = 0
        self.output_folder = output_folder

    def on_tick(self, sim):
        fname = '%s/%05d.png' % (self.output_folder, self.frame_index)
        sim.take_screenshot(fname)
        self.frame_index += 1

    def wait_for(self, wait_time, sim):
        """
        Pause the video by repeating the last frame for a certain duration.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        wait_time : scalar
            Duration in [s].
        """
        for _ in xrange(int(wait_time / sim.dt)):
            self.on_tick(sim)
