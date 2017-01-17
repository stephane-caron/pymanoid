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

from numpy import hstack, zeros
from threading import Lock
from time import time

from draw import draw_line, draw_polygon, draw_polyhedron
from draw import draw_force, draw_wrench
from misc import norm


class Process(object):

    """
    Processes implement the ``on_tick`` method called by the simulation.
    """

    def __init__(self):
        self.paused = False
        self._log_comp_times = False

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
        raise NotImplementedError

    def pause(self):
        """Stop calling the process at new clock ticks."""
        self.paused = True

    def resume(self):
        """Resume calling the process at new clock ticks."""
        self.paused = False


class PointMassForceDrawer(Process):

    """
    Draw contact forces for a point-mass system in multi-contact.

    Parameters
    ----------
    pm : PointMass
        Point-mass to which forces are applied.
    cs : ContactSet
        Set of contacts providing interaction forces.
    scale : scalar
        Force-to-distance conversion ratio in [m] / [N].
    """

    KO_COLOR = [.8, .4, .4]

    def __init__(self, pm, cs, scale=0.0025):
        super(PointMassForceDrawer, self).__init__()
        self.cs = cs
        self.handles = []
        self.last_bkgnd_switch = None
        self.pm = pm
        self.scale = scale

    def on_tick(self, sim):
        """Find supporting contact forces at each COM acceleration update."""
        p, mass = self.pm.p, self.pm.mass
        pdd = self.pm.pdd  # needs to be stored by the user
        gravity = sim.gravity
        wrench = hstack([mass * (pdd - gravity), zeros(3)])
        contact_set = self.cs() if callable(self.cs) else self.cs
        support = contact_set.find_supporting_forces(wrench, p)
        if not support:
            self.handles = []
            sim.viewer.SetBkgndColor(self.KO_COLOR)
            self.last_bkgnd_switch = time()
        else:
            self.handles = [
                draw_force(c, fc, self.scale) for (c, fc) in support]
        if self.last_bkgnd_switch is not None \
                and time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            sim.viewer.SetBkgndColor(sim.BACKGROUND_COLOR)
            self.last_bkgnd_switch = None


class PointMassWrenchDrawer(PointMassForceDrawer):

    """
    Draw contact wrenches for a point-mass system in multi-contact.

    Parameters
    ----------
    pm : PointMass
        Point-mass to which forces are applied.
    cs : ContactSet
        Set of contacts providing interaction forces.
    scale : scalar
        Force-to-distance conversion ratio in [m] / [N].
    """

    def on_tick(self, sim):
        """Find supporting contact forces at each COM acceleration update."""
        p, mass = self.pm.p, self.pm.mass
        pdd = self.pm.pdd  # needs to be stored by the user
        gravity = sim.gravity
        wrench = hstack([mass * (pdd - gravity), zeros(3)])
        contact_set = self.cs() if callable(self.cs) else self.cs
        support = contact_set.find_supporting_wrenches(wrench, p)
        if not support:
            self.handles = []
            sim.viewer.SetBkgndColor(self.KO_COLOR)
            self.last_bkgnd_switch = time()
        else:
            self.handles = [
                draw_wrench(contact, w_c, scale=self.scale)
                for (contact, w_c) in support]
        if self.last_bkgnd_switch is not None \
                and time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            sim.viewer.SetBkgndColor(sim.BACKGROUND_COLOR)
            self.last_bkgnd_switch = None


class PreviewBuffer(Process):

    """
    Buffer to store controls output by a preview controller.

    Parameters
    ----------
    callback : function
        Function to call with each new control `(u, dT)`.
    """

    def __init__(self, callback):
        super(PreviewBuffer, self).__init__()
        self.callback = callback
        self.cur_control = None
        self.preview = None
        self.preview_index = 0
        self.preview_lock = Lock()
        self.rem_time = 0.

    def update_preview(self, preview):
        """
        Update preview with a filled PreviewControl object.

        Parameters
        ----------
        preview : PreviewControl
            New PreviewControl instance to store into the buffer.
        """
        with self.preview_lock:
            self.preview_index = 0
            self.preview = preview

    def get_next_control(self):
        """
        Return the next pair ``(u, dT)`` in the preview window.
        """
        with self.preview_lock:
            if self.preview is None:
                return (zeros(3), 0.)
            j = 3 * self.preview_index
            u = self.preview.U[j:j + 3]
            if u.shape[0] == 0:
                self.preview = None
                return (zeros(3), 0.)
            dT = self.preview.timestep
            self.preview_index += 1
            return (u, dT)

    def on_tick(self, sim):
        """
        Entry point called at each simulation tick.

        Parameters
        ----------
        sim : Simulation
            Current simulation instance.
        """
        if self.rem_time < sim.dt:
            u, dT = self.get_next_control()
            self.cur_control = u
            self.rem_time = dT
        self.callback(self.cur_control, sim.dt)
        self.rem_time -= sim.dt


class SupportAreaDrawer(Process):

    """Draw a given support area of a contact set."""

    def __init__(self, contact_set, z=0.):
        super(SupportAreaDrawer, self).__init__()
        contact_dict = contact_set.contact_dict
        self.contact_dict = contact_dict
        self.contact_poses = {}
        self.contact_set = contact_set
        self.handle = None
        self.z = z
        self.update_contact_poses()
        self.update_polygon()

    def on_tick(self, sim):
        for (k, c) in self.contact_dict.iteritems():
            if norm(c.pose - self.contact_poses[k]) > 1e-10:
                self.update_contact_poses()
                self.update_polygon()
                break

    def update_contact_poses(self):
        for (k, c) in self.contact_dict.iteritems():
            self.contact_poses[k] = c.pose

    def update_polygon(self):
        raise NotImplementedError


class SEPDrawer(SupportAreaDrawer):

    def update_polygon(self):
        self.handle = None
        try:
            vertices = self.contact_set.compute_static_equilibrium_polygon()
            self.handle = draw_polygon(
                [(x[0], x[1], self.z) for x in vertices],
                normal=[0, 0, 1], color=(0.5, 0., 0.5, 0.5))
        except Exception as e:
            print "SEPDrawer:", e


class StaticForceDrawer(Process):

    KO_COLOR = [.8, .4, .4]
    OK_COLOR = [1., 1., 1.]

    def __init__(self, com, contact_set, force_scale=0.0025):
        super(StaticForceDrawer, self).__init__()
        self.com = com
        self.contact_set = contact_set
        self.force_scale = force_scale
        self.handles = []
        self.last_bkgnd_switch = None

    def on_tick(self, sim):
        """Find supporting contact forces at each COM acceleration update."""
        support = self.contact_set.find_static_supporting_forces(
            self.com.p, self.com.mass)
        if not support:
            self.handles = []
            sim.viewer.SetBkgndColor(self.KO_COLOR)
            self.last_bkgnd_switch = time()
        else:
            self.handles = [
                draw_force(c, fc, self.force_scale) for (c, fc) in support]
        if self.last_bkgnd_switch is not None \
                and time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            sim.viewer.SetBkgndColor(self.OK_COLOR)
            self.last_bkgnd_switch = None


class TrajectoryDrawer(Process):

    def __init__(self, body, combined='b-', color=None, linewidth=3,
                 linestyle=None):
        super(TrajectoryDrawer, self).__init__()
        color = color if color is not None else combined[0]
        linestyle = linestyle if linestyle is not None else combined[1]
        assert linestyle in ['-', '.']
        self.body = body
        self.color = color
        self.handles = []
        self.last_pos = body.p
        self.linestyle = linestyle
        self.linewidth = linewidth

    def on_tick(self, sim):
        if self.linestyle == '-':
            self.handles.append(draw_line(
                self.last_pos, self.body.p, color=self.color,
                linewidth=self.linewidth))
        self.last_pos = self.body.p

    def dash_graph_handles(self):
        for i in xrange(len(self.handles)):
            if i % 2 == 0:
                self.handles[i] = None


class ZMPSupportAreaDrawer(SupportAreaDrawer):

    """Draw the pendular ZMP support area of a contact set."""

    def __init__(self, stance, z=0.):
        self.last_com = stance.com.p
        self.method = 'cdd'
        self.stance = stance
        super(ZMPSupportAreaDrawer, self).__init__(stance, z)

    def on_tick(self, sim):
        super(ZMPSupportAreaDrawer, self).on_tick(sim)
        if norm(self.stance.com.p - self.last_com) > 1e-10:
            self.update_contact_poses()
            self.update_polygon()
            self.last_com = self.stance.com.p

    def update_polygon(self):
        self.handle = None
        try:
            vertices = self.contact_set.compute_zmp_support_area(
                self.stance.com.p, [0, 0, self.z], method=self.method)
            self.handle = draw_polygon(
                [(x[0], x[1], self.z) for x in vertices],
                normal=[0, 0, 1], color=(0.0, 0.5, 0.5, 0.5))
        except Exception as e:
            print "ZMPSupportAreaDrawer:", e


class COMAccelConeDrawer(ZMPSupportAreaDrawer):

    def update_polygon(self):
        self.handle = None
        try:
            cone = self.contact_set.compute_pendular_accel_cone(
                self.stance.com.p)
            vscale = [self.stance.com.p + 0.1 * acc for acc in cone.vertices]
            self.handle = draw_polyhedron(vscale, 'r.-#')
        except Exception as e:
            print "COMAccelConeDrawer:", e
