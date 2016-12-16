#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2016 Stephane Caron <stephane.caron@normalesup.org>
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

import IPython
import os
import sys
import time

from numpy import arange, array, bmat, cross, dot, eye, hstack, vstack, zeros
from numpy import cos, pi, sin
from numpy.random import random, seed
from scipy.linalg import block_diag
from warnings import warn

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../')
    import pymanoid

from pymanoid import Contact, ContactSet, PointMass, Polytope, Stance
from pymanoid.draw import draw_force, draw_line, draw_point, draw_points
from pymanoid.draw import draw_polyhedron, draw_polygon
from pymanoid.misc import normalize
from pymanoid.mpc import PreviewBuffer
try:
    from pymanoid.mpc import VSPreviewControl as PreviewControl
    print "using VSPreviewController"
except ImportError:
    from pymanoid.mpc import PreviewControl as PreviewControl
    print "using default PreviewController"
from pymanoid.polyhedra import intersect_polygons
from pymanoid.process import Process, TrajectoryDrawer
from pymanoid.robots import JVRC1
from pymanoid.tasks import DOFTask, MinCAMTask
from pymanoid.walking import SwingFoot, WalkingFSM


def generate_staircase(radius, angular_step, height, roughness, friction,
                       ds_duration, ss_duration, init_com_offset=None):
    """
    Generate a new slanted staircase with tilted steps.

    INPUT:

    - ``radius`` -- staircase radius (in [m])
    - ``angular_step`` -- angular step between contacts (in [rad])
    - ``height`` -- altitude variation (in [m])
    - ``roughness`` -- amplitude of contact roll, pitch and yaw (in [rad])
    - ``friction`` -- friction coefficient between a robot foot and a step
    - ``ds_duration`` -- duration of double-support phases in [s]
    - ``ss_duration`` -- duration of single-support phases in [s]
    - ``init_com_offset`` -- (optional) initial offset applied to first stance
    """
    stances = []
    contact_shape = (0.12, 0.06)
    first_left_foot = None
    prev_right_foot = None
    for theta in arange(0., 2 * pi, angular_step):
        left_foot = Contact(
            shape=contact_shape,
            pos=[radius * cos(theta),
                 radius * sin(theta),
                 radius + .5 * height * sin(theta)],
            rpy=(roughness * (random(3) - 0.5) + [0, 0, theta + .5 * pi]),
            static_friction=friction,
            visible=True)
        if first_left_foot is None:
            first_left_foot = left_foot
        right_foot = Contact(
            shape=contact_shape,
            pos=[1.2 * radius * cos(theta + .5 * angular_step),
                 1.2 * radius * sin(theta + .5 * angular_step),
                 radius + .5 * height * sin(theta + .5 * angular_step)],
            rpy=(roughness * (random(3) - 0.5) + [0, 0, theta + .5 * pi]),
            static_friction=friction,
            visible=True)
        if prev_right_foot is not None:
            com_target = left_foot.p + [0., 0., JVRC1.leg_length]
            stances.append(Stance(
                com_target, left_foot=left_foot, right_foot=prev_right_foot,
                label='DS-L', duration=ds_duration))
            stances.append(Stance(
                com_target, left_foot=left_foot, label='SS-L',
                duration=ss_duration))
        com_target = right_foot.p + [0., 0., JVRC1.leg_length]
        if init_com_offset is not None:
            com_target += init_com_offset
            init_com_offset = None
        stances.append(Stance(
            com_target, left_foot=left_foot, right_foot=right_foot,
            label='DS-R', duration=ds_duration))
        stances.append(Stance(
            com_target, right_foot=right_foot, label='SS-R',
            duration=ss_duration))
        prev_right_foot = right_foot
    com_target = first_left_foot.p + [0., 0., JVRC1.leg_length]
    stances.append(Stance(
        com_target, left_foot=first_left_foot, right_foot=prev_right_foot,
        label='DS-L', duration=ds_duration))
    stances.append(Stance(
        com_target, left_foot=first_left_foot, label='SS-L',
        duration=ss_duration))
    return stances


class COMTube(object):

    """
    Primal tube of COM locations computed along with its dual acceleration cone.

    When there is an SS-to-DS contact switch, this strategy computes one primal
    tube and two dual intersection cones. The primal tube is a parallelepiped
    containing both the COM current and target locations. Its dual cone is used
    during the DS phase. The dual cone for the SS phase is calculated by
    intersecting the latter with the dual cone of the current COM position in
    single-contact.
    """

    def __init__(self, start_com, target_com, start_stance, next_stance, radius,
                 margin=0.01):
        """
        Create a new COM trajectory tube.

        INPUT:

        - ``start_com`` -- start position of the COM
        - ``target_com`` -- end position of the COM
        - ``start_stance`` -- stance used to compute the contact wrench cone
        - ``radius`` -- side of the cross-section square (for ``shape`` > 2)
        - ``margin`` -- safety margin (in [m]) around boundary COM positions
        """
        self.dual_hrep = []
        self.dual_vrep = []
        self.margin = margin
        self.next_stance = next_stance
        self.primal_hrep = []
        self.primal_vrep = []
        self.radius = radius
        self.start_com = start_com
        self.start_stance = start_stance
        self.target_com = target_com

    def compute_primal_vrep(self):
        """Compute vertices of the primal tube."""
        delta = self.target_com - self.start_com
        n = normalize(delta)
        t = array([0., 0., 1.])
        t -= dot(t, n) * n
        t = normalize(t)
        b = cross(n, t)
        cross_section = [dx * t + dy * b for (dx, dy) in [
            (+self.radius, +self.radius),
            (+self.radius, -self.radius),
            (-self.radius, +self.radius),
            (-self.radius, -self.radius)]]
        tube_start = self.start_com - self.margin * n
        tube_end = self.target_com + self.margin * n
        vertices = (
            [tube_start + s for s in cross_section] +
            [tube_end + s for s in cross_section])
        self.full_vrep = vertices
        if self.start_stance.label.startswith('SS'):
            if all(abs(self.start_stance.com.p - self.target_com) < 1e-3):
                self.primal_vrep = [vertices]
            else:  # beginning of SS phase
                self.primal_vrep = [
                    [self.start_com],  # single-support
                    vertices]          # ensuing double-support
        else:  # start_stance is DS
            self.primal_vrep = [
                vertices,             # double-support
                [self.target_com]]    # final single-support

    def compute_primal_hrep(self):
        """Compute halfspaces of the primal tube."""
        try:
            self.full_hrep = (Polytope.hrep(self.full_vrep))
        except RuntimeError as e:
            raise Exception("Could not compute primal hrep: %s" % str(e))

    def compute_dual_vrep(self):
        """Compute vertices of the dual cones."""
        if len(self.primal_vrep) == 1:
            dual = self.start_stance.compute_pendular_accel_cone(
                com=self.primal_vrep[0])
            self.dual_vrep = [dual.vertices]
            return
        # now, len(self.primal_vrep) == 2
        ds_then_ss = len(self.primal_vrep[0]) > 1
        if ds_then_ss:
            ds_vertices_2d = self.start_stance.compute_pendular_accel_cone(
                com=self.full_vrep, reduced=True)
            ss_vertices_2d = self.next_stance.compute_pendular_accel_cone(
                com=self.primal_vrep[1], reduced=True)
        else:  # SS, then DS
            ss_vertices_2d = self.start_stance.compute_pendular_accel_cone(
                com=self.primal_vrep[0], reduced=True)
            ds_vertices_2d = self.next_stance.compute_pendular_accel_cone(
                com=self.full_vrep, reduced=True)
        ss_vertices_2d = intersect_polygons(ds_vertices_2d, ss_vertices_2d)
        ds_cone = ContactSet._expand_reduced_pendular_cone(ds_vertices_2d)
        ss_cone = ContactSet._expand_reduced_pendular_cone(ss_vertices_2d)
        if ds_then_ss:
            self.dual_vrep = [ds_cone.vertices, ss_cone.vertices]
        else:  # SS, then DS
            self.dual_vrep = [ss_cone.vertices, ds_cone.vertices]

    def compute_dual_hrep(self):
        """Compute halfspaces of the dual cones."""
        for (stance_id, cone_vertices) in enumerate(self.dual_vrep):
            B, c = Polytope.hrep(cone_vertices)
            self.dual_hrep.append((B, c))


class COMTubePreviewControl(Process):

    def __init__(self, com, fsm, preview_buffer, nb_mpc_steps, tube_radius):
        """
        Create a new feedback controller that continuously runs the preview
        controller and sends outputs to a COMAccelBuffer.

        INPUT:

        - ``com`` -- PointMass containing current COM state
        - ``fsm`` -- instance of finite state machine
        - ``preview_buffer`` -- PreviewBuffer to send MPC outputs to
        - ``nb_mpc_steps`` -- discretization step of the preview window
        - ``tube_radius`` -- tube radius (in L1 norm)
        """
        super(COMTubePreviewControl, self).__init__()
        self.com = com
        self.fsm = fsm
        self.last_phase_id = -1
        self.nb_mpc_steps = nb_mpc_steps
        self.preview_buffer = preview_buffer
        self.preview_control = None
        self.target_com = PointMass(fsm.cur_stance.com.p, 30., color='g')
        self.tube = None
        self.tube_radius = tube_radius

    def on_tick(self, sim):
        """
        Entry point called at each simulation tick.
        """
        preview_targets = self.fsm.get_preview_targets()
        switch_time, horizon, target_com, target_comd = preview_targets
        self.target_com.set_pos(target_com)
        self.target_com.set_velocity(target_comd)
        try:
            self.compute_preview_tube()
        except Exception as e:
            print "Tube error: %s" % str(e)
            return
        try:
            self.compute_preview_control(switch_time, horizon)
        except Exception as e:
            print "PreviewControl error: %s" % str(e)
            return
        sim.log_comp_time(
            'qp_solve', self.preview_control.solve_time)
        sim.log_comp_time(
            'qp_solve_and_build', self.preview_control.solve_and_build_time)

    def compute_preview_tube(self):
        """Compute preview tube and store it in ``self.tube``."""
        cur_com, target_com = self.com.p, self.target_com.p
        cur_stance = self.fsm.cur_stance
        next_stance = self.fsm.next_stance
        self.tube = COMTube(
            cur_com, target_com, cur_stance, next_stance, self.tube_radius)
        t0 = time.time()
        self.tube.compute_primal_vrep()
        t1 = time.time()
        self.tube.compute_primal_hrep()
        t2 = time.time()
        self.tube.compute_dual_vrep()
        t3 = time.time()
        self.tube.compute_dual_hrep()
        t4 = time.time()
        sim.log_comp_time('tube_primal_vrep', t1 - t0)
        sim.log_comp_time('tube_primal_hrep', t2 - t1)
        sim.log_comp_time('tube_dual_vrep', t3 - t2)
        sim.log_comp_time('tube_dual_hrep', t4 - t3)

    def compute_preview_control(self, switch_time, horizon,
                                state_constraints=False):
        """Compute controller and store it in ``self.preview_control``."""
        cur_com = self.com.p
        cur_comd = self.com.pd
        target_com = self.target_com.p
        target_comd = self.target_com.pd
        dT = horizon / self.nb_mpc_steps
        I = eye(3)
        A = array(bmat([[I, dT * I], [zeros((3, 3)), I]]))
        B = array(bmat([[.5 * dT ** 2 * I], [dT * I]]))
        x_init = hstack([cur_com, cur_comd])
        x_goal = hstack([target_com, target_comd])
        switch_step = int(switch_time / dT)
        G_list = []
        h_list = []
        C1, d1 = self.tube.dual_hrep[0]
        E, f = None, None
        if state_constraints:
            E, f = self.tube.full_hrep
        if 0 <= switch_step < self.nb_mpc_steps - 1:
            C2, d2 = self.tube.dual_hrep[1]
        for k in xrange(self.nb_mpc_steps):
            if k <= switch_step:
                G_list.append(C1)
                h_list.append(d1)
            else:  # k > switch_step
                G_list.append(C2)
                h_list.append(d2)
        G = block_diag(*G_list)
        h = hstack(h_list)
        self.preview_control = PreviewControl(
            A, B, G, h, x_init, x_goal, self.nb_mpc_steps, E, f)
        self.preview_control.switch_step = switch_step
        self.preview_control.timestep = dT
        self.preview_control.compute_dynamics()
        try:
            self.preview_control.compute_control()
            self.preview_buffer.update_preview(self.preview_control)
        except ValueError:
            print "MPC couldn't solve QP, constraints may be inconsistent"


class ForceDrawer(Process):

    """
    Draw supporting contact forces at each simulation tick.
    """

    def __init__(self):
        super(ForceDrawer, self).__init__()
        self.force_scale = 0.0025
        self.handles = []
        self.last_bkgnd_switch = None
        self.mass = robot.mass

    def on_tick(self, sim):
        """Entry point called at each simulation tick."""
        comdd = preview_buffer.cur_control
        wrench = hstack([self.mass * (comdd - sim.gravity), zeros(3)])
        support = fsm.cur_stance.find_supporting_forces(
            wrench, com_target.p, self.mass, 10.)
        if not support:
            self.handles = []
            sim.viewer.SetBkgndColor([.8, .4, .4])
            self.last_bkgnd_switch = time.time()
        else:
            self.handles = [draw_force(c, fc, self.force_scale)
                            for (c, fc) in support]
        if self.last_bkgnd_switch is not None \
                and time.time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            sim.viewer.SetBkgndColor([.6, .6, .8])
            self.last_bkgnd_switch = None


class PreviewDrawer(pymanoid.Process):

    """
    Draw preview trajectory, in blue and yellow for the SS and DS parts
    respectively.
    """

    def __init__(self):
        super(PreviewDrawer, self).__init__()
        self.draw_free_traj = False
        self.handles = []

    def on_tick(self, sim):
        if preview_buffer.preview is None:
            return
        com_pre, comd_pre = com_target.p, com_target.pd
        com_free, comd_free = com_target.p, com_target.pd
        dT = preview_buffer.preview.timestep
        self.handles = []
        self.handles.append(
            draw_point(com_target.p, color='m', pointsize=0.007))
        for preview_index in xrange(len(preview_buffer.preview.U) / 3):
            com_pre0 = com_pre
            j = 3 * preview_index
            comdd = preview_buffer.preview.U[j:j + 3]
            com_pre = com_pre + comd_pre * dT + comdd * .5 * dT ** 2
            comd_pre += comdd * dT
            color = \
                'b' if preview_index <= preview_buffer.preview.switch_step \
                else 'y'
            self.handles.append(
                draw_point(com_pre, color=color, pointsize=0.005))
            self.handles.append(
                draw_line(com_pre0, com_pre, color=color, linewidth=3))
            if self.draw_free_traj:
                com_free0 = com_free
                com_free = com_free + comd_free * dT
                self.handles.append(
                    draw_point(com_free, color='g', pointsize=0.005))
                self.handles.append(
                    draw_line(com_free0, com_free, color='g', linewidth=3))


class TubeDrawer(pymanoid.Process):

    """
    Draw preview COM tube along with its dual acceleration cone.
    """

    def __init__(self):
        super(TubeDrawer, self).__init__()
        self.comdd_handle = []
        self.cone_handles = []
        self.poly_handles = []
        self.acc_scale = 0.1
        self.trans = array([0., 0., 1.1])

    def on_tick(self, sim):
        try:
            self.draw_primal(mpc.tube)
        except Exception as e:
            print "Drawing of polytopes failed: %s" % str(e)
        try:
            self.draw_dual(mpc.tube)
        except Exception as e:
            print "Drawing of dual cones failed: %s" % str(e)
        if True:
            self.draw_comdd()

    def draw_primal(self, tube):
        self.poly_handles = []
        colors = [(0.5, 0.5, 0., 0.3), (0., 0.5, 0.5, 0.3)]
        if tube.start_stance.label.startswith('SS'):
            colors.reverse()
        for (i, vertices) in enumerate(tube.primal_vrep):
            color = colors[i]
            if len(vertices) == 1:
                self.poly_handles.append(
                    draw_point(vertices[0], color=color, pointsize=0.01))
            else:
                self.poly_handles.extend(
                    draw_polyhedron(vertices, '*.-#', color=color))

    def draw_dual(self, tube):
        self.cone_handles = []
        self.trans = com_target.p
        apex = [0., 0., self.acc_scale * -9.81] + self.trans
        colors = [(0.5, 0.5, 0., 0.3), (0., 0.5, 0.5, 0.3)]
        if tube.start_stance.label.startswith('SS'):
            colors.reverse()
        for (stance_id, cone_vertices) in enumerate(tube.dual_vrep):
            color = colors[stance_id]
            vscale = [self.acc_scale * array(v) + self.trans
                      for v in cone_vertices]
            self.cone_handles.extend(
                self.draw_cone_fast(
                    apex=apex, axis=[0, 0, 1], section=vscale[1:],
                    combined='r-#', color=color))

    def draw_comdd(self):
        comdd = self.acc_scale * preview_buffer.cur_control + self.trans
        self.comdd_handle = [
            draw_line(self.trans, comdd, color='r', linewidth=3),
            draw_points([self.trans, comdd], color='r', pointsize=0.005)]

    def draw_cone_fast(self, apex, axis, section, combined='r-#', color=None,
                       linewidth=2., pointsize=0.05):
        """
        Draw a 3D cone defined from its apex, axis vector and a cross-section
        polygon (defined in the plane orthogonal to the axis vector).

        INPUT:

        - ``apex`` -- position of the origin of the cone in world coordinates
        - ``axis`` -- unit vector directing the cone axis and lying inside
        - ``combined`` -- (default: 'g-#') drawing spec in matplotlib fashion
        - ``linewidth`` -- thickness of the edges of the cone
        - ``pointsize`` -- point size in meters

        OUTPUT:

        A list of OpenRAVE handles. Must be stored in some variable, otherwise
        the drawn object will vanish instantly.
        """
        if len(section) < 1:
            warn("Trying to draw an empty cone")
            return []
        from pymanoid.draw import matplotlib_to_rgba
        color = color if color is not None else matplotlib_to_rgba(combined[0])
        handles = draw_polygon(
            points=section, normal=axis, combined=combined, color=color)
        edges = vstack([[apex, vertex] for vertex in section])
        edges = array(edges)
        edge_color = array(color) * 0.7
        edge_color[3] = 1.
        handles.append(sim.env.drawlinelist(
            edges, linewidth=linewidth, colors=edge_color))
        return handles


if __name__ == "__main__":
    seed(42)
    sim = pymanoid.Simulation(dt=0.03)
    robot = JVRC1(download_if_needed=True)
    sim.set_viewer()
    robot.set_transparency(0.3)

    staircase = generate_staircase(
        radius=1.4,
        angular_step=0.5,
        height=1.2,
        roughness=0.5,
        friction=0.7,
        ds_duration=0.7,
        ss_duration=1.0,
        init_com_offset=array([0., 0., 0.]))

    com_target = PointMass([0, 0, 0], 20.)
    preview_buffer = PreviewBuffer(
        callback=lambda u, dT: com_target.integrate_acceleration(u, dT))
    swing_foot = SwingFoot(swing_height=0.15)
    fsm = WalkingFSM(staircase, robot, swing_foot, cycle=True)

    mpc = COMTubePreviewControl(
        com_target, fsm, preview_buffer,
        nb_mpc_steps=10,
        tube_radius=0.01)

    robot.init_ik(robot.whole_body)
    robot.set_ff_pos([0, 0, 2])  # start IK with the robot above contacts
    robot.generate_posture(fsm.cur_stance, max_it=50)

    com_target.set_pos(robot.com)
    robot.ik.tasks['com'].update_target(com_target)
    robot.ik.add_task(DOFTask(robot, robot.WAIST_P, 0.2, weight=1e-3))
    robot.ik.add_task(DOFTask(robot, robot.WAIST_Y, 0., weight=1e-3))
    robot.ik.add_task(DOFTask(robot, robot.WAIST_R, 0., weight=1e-3))
    robot.ik.add_task(DOFTask(robot, robot.ROT_P, 0., weight=1e-3))
    robot.ik.add_task(DOFTask(robot, robot.R_SHOULDER_R, -0.5, weight=1e-3))
    robot.ik.add_task(DOFTask(robot, robot.L_SHOULDER_R, 0.5, weight=1e-3))
    robot.ik.add_task(MinCAMTask(robot, weight=1e-4))
    robot.ik.tasks['posture'].weight = 1e-5

    sim.schedule(fsm)
    sim.schedule(mpc)
    sim.schedule(preview_buffer)
    sim.schedule(robot.ik_process)

    com_traj_drawer = TrajectoryDrawer(com_target, 'b-')
    force_drawer = ForceDrawer()
    lf_traj_drawer = TrajectoryDrawer(robot.left_foot, 'g-')
    preview_drawer = PreviewDrawer()
    rf_traj_drawer = TrajectoryDrawer(robot.right_foot, 'r-')
    tube_drawer = TubeDrawer()

    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(force_drawer)
    sim.schedule_extra(lf_traj_drawer)
    sim.schedule_extra(preview_drawer)
    sim.schedule_extra(rf_traj_drawer)
    sim.schedule_extra(tube_drawer)

    print """

Multi-contact Walking Pattern Generation
========================================

Ready to go! You can control the simulation by:

    sim.start() -- run/resume simulation in a separate thread
    sim.step(100) -- run simulation in current thread for 100 steps
    sim.stop() -- stop/pause simulation

You can access all state variables via this IPython shell.
Here is the list of global objects. Use <TAB> to see what's inside.

    fsm -- finite state machine
    mpc -- model-preview controller
    preview_buffer -- stores MPC output and feeds it to the IK
    robot -- kinematic model of the robot (includes IK solver)

Enjoy :)

"""
    if IPython.get_ipython() is None:
        IPython.embed()
