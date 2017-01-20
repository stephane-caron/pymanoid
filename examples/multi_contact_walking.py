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
from pymanoid.body import Box
from pymanoid.draw import draw_line, draw_point, draw_points
from pymanoid.draw import draw_polyhedron, draw_polygon
from pymanoid.misc import interpolate_pose_linear, normalize
from pymanoid.mpc import PreviewBuffer
from pymanoid.polyhedra import intersect_polygons
from pymanoid.drawers import TrajectoryDrawer, PointMassWrenchDrawer
from pymanoid.robots import JVRC1
from pymanoid.rotations import quat_slerp, rotation_matrix_from_quat
from pymanoid.tasks import ContactTask, DOFTask, LinkPoseTask, MinCAMTask

try:
    from pymanoid.mpc import VSPreviewControl as PreviewControl
    print "using VSPreviewController"
except ImportError:
    from pymanoid.mpc import PreviewControl as PreviewControl
    print "using default PreviewController"


def generate_staircase(radius, angular_step, height, roughness, friction,
                       ds_duration, ss_duration, init_com_offset=None):
    """
    Generate a new slanted staircase with tilted steps.

    Parameters
    ----------
    radius : scalar
        Staircase radius in [m].
    angular_step : scalar
        Angular step between contacts in [rad].
    height : scalar
        Altitude variation in [m].
    roughness : scalar
        Amplitude of contact roll, pitch and yaw in [rad].
    friction : scalar
        Friction coefficient between a robot foot and a step.
    ds_duration : scalar
        Duration of double-support phases in [s].
    ss_duration : scalar
        Duration of single-support phases in [s].
    init_com_offset : array, optional
        Initial offset applied to first stance.
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


class SwingFoot(Box):

    """
    Invisible body used for swing foot interpolation.

    Parameters
    ----------
    swing_height : double
        Height in [m] for the apex of the foot trajectory.
    color : char, optional
        Color applied to all links of the KinBody.
    visible : bool, optional
        Initial visibility.
    transparency : double, optional
        Transparency value from 0 (opaque) to 1 (invisible).
    """

    THICKNESS = 0.01

    def __init__(self, swing_height, color='c', visible=False,
                 transparency=0.5):
        super(SwingFoot, self).__init__(
            X=0.12, Y=0.06, Z=self.THICKNESS, color=color, visible=visible,
            transparency=transparency, dZ=-self.THICKNESS)
        self.end_pose = None
        self.mid_pose = None
        self.start_pose = None
        self.swing_height = swing_height

    def reset(self, start_pose, end_pose):
        """
        Reset both end poses of the interpolation.

        Parameters
        ----------
        start_pose : array
            New start pose.
        end_pose : array
            New end pose.
        """
        mid_pose = interpolate_pose_linear(start_pose, end_pose, .5)
        mid_n = rotation_matrix_from_quat(mid_pose[:4])[0:3, 2]
        mid_pose[4:] += self.swing_height * mid_n
        self.set_pose(start_pose)
        self.start_pose = start_pose
        self.mid_pose = mid_pose
        self.end_pose = end_pose

    def update_pose(self, s):
        """
        Update pose to a given index ``s`` in the swing-foot motion.

        Parameters
        ----------
        s : scalar
            Index between 0 and 1 in the swing-foot motion.
        """
        if s >= 1.:
            return
        elif s <= .5:
            pose0 = self.start_pose
            pose1 = self.mid_pose
            y = 2. * s
        else:  # .5 < x < 1
            pose0 = self.mid_pose
            pose1 = self.end_pose
            y = 2. * s - 1.
        pos = (1. - y) * pose0[4:] + y * pose1[4:]
        quat = quat_slerp(pose0[:4], pose1[:4], y)
        self.set_pose(hstack([quat, pos]))


class WalkingFSM(pymanoid.Process):

    """
    Finite State Machine for biped walking.
    """

    def __init__(self, stances, robot, swing_height, cycle=False):
        """
        Create a new finite state machine.

        Parameters
        ----------
        stances : list of Stances
            Consecutives stances traversed by the FSM.
        robot : Robot
            Controller robot.
        swing_height : scalar
            Relative height in [m] for the apex of swing foot trajectories.
        cycle : bool, optional
            If ``True``, the first stance will succeed the last one.
        """
        super(WalkingFSM, self).__init__()
        self.cur_phase = stances[0].label
        self.cur_stance = stances[0]
        self.cur_stance_id = 0
        self.cycle = cycle
        self.is_over = False
        self.nb_stances = len(stances)
        self.rem_time = stances[0].duration
        self.robot = robot
        self.stances = stances
        self.swing_foot = SwingFoot(swing_height)
        self.verbose = True

    @property
    def next_stance(self):
        next_stance_id = self.cur_stance_id + 1
        if next_stance_id >= self.nb_stances:
            next_stance_id = 0 if self.cycle else self.cur_stance_id
        return self.stances[next_stance_id]

    @property
    def next_next_stance(self):
        next_next_stance_id = self.cur_stance_id + 2
        if next_next_stance_id >= self.nb_stances:
            if self.cycle:
                next_next_stance_id %= self.nb_stances
            else:  # not self.cycle:
                next_next_stance_id = self.nb_stances - 1
        return self.stances[next_next_stance_id]

    def get_cur_stance(self):
        """Used by the PointMassWrenchDrawer process."""
        return self.cur_stance

    def get_preview_targets(self):
        stance_foot = self.cur_stance.left_foot \
            if self.cur_stance.label.endswith('L') else \
            self.cur_stance.right_foot
        if self.cur_stance.label.startswith('SS') \
                and self.rem_time < 0.5 * self.cur_stance.duration:
            horizon = self.rem_time \
                + self.next_stance.duration \
                + 0.5 * self.next_next_stance.duration
            target_com = self.next_stance.com.p
            target_comd = (target_com - self.cur_stance.com.p) / horizon
        elif self.cur_stance.label.startswith('DS'):
            horizon = self.rem_time + 0.5 * self.next_stance.duration
            target_com = self.cur_stance.com.p
            target_comd = 0.4 * stance_foot.t
        else:  # single support with plenty of time ahead
            horizon = self.rem_time
            target_com = self.cur_stance.com.p
            target_comd = 0.4 * stance_foot.t
        return (self.rem_time, horizon, target_com, target_comd)

    def on_tick(self, sim):
        """
        Update the FSM after a tick of the control loop.

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
        """
        if self.is_over:
            return

        def can_switch_to_ss():
            com = self.robot.com
            dist_inside_sep = self.next_stance.dist_to_sep_edge(com)
            return dist_inside_sep > -0.15

        if self.rem_time > 0.:
            if self.cur_stance.label.startswith('SS'):
                progress = 1. - self.rem_time / self.cur_stance.duration
                self.swing_foot.update_pose(progress)
            self.rem_time -= sim.dt
        elif (self.cur_stance.label.startswith('DS') and not
              can_switch_to_ss()):
            print "FSM: not ready for single-support yet..."
        elif self.cur_stance_id == self.nb_stances - 1 and not self.cycle:
            self.is_over = True
        else:
            # NB: in the following block, cur_stance has not been updated yet
            if self.cur_stance.label == 'DS-R' \
                    and self.next_next_stance.label == 'DS-L':
                self.swing_foot.reset(
                    self.cur_stance.left_foot.pose,
                    self.next_next_stance.left_foot.pose)
            elif self.cur_stance.label == 'DS-L' \
                    and self.next_next_stance.label == 'DS-R':
                self.swing_foot.reset(
                    self.cur_stance.right_foot.pose,
                    self.next_next_stance.right_foot.pose)
            # now that we have read swing foot poses, we update cur_stance
            self.cur_stance_id = (self.cur_stance_id + 1) % self.nb_stances
            self.cur_stance = self.stances[self.cur_stance_id]
            self.rem_time = self.cur_stance.duration
            self.update_robot_ik()
            if self.verbose:
                print "FSM switched to '%s' stance" % self.cur_stance.label

    def update_robot_ik(self):
        prev_lf_task = self.robot.ik.get_task(self.robot.left_foot.name)
        prev_rf_task = self.robot.ik.get_task(self.robot.right_foot.name)
        contact_weight = max(prev_lf_task.weight, prev_rf_task.weight)
        swing_weight = 1e-1
        self.robot.ik.remove_task(self.robot.left_foot.name)
        self.robot.ik.remove_task(self.robot.right_foot.name)
        if self.cur_stance.left_foot is not None:
            left_foot_task = ContactTask(
                self.robot, self.robot.left_foot, self.cur_stance.left_foot,
                weight=contact_weight)
        else:  # left_foot is swinging
            left_foot_task = LinkPoseTask(
                self.robot, self.robot.left_foot, self.swing_foot,
                weight=swing_weight)
        if self.cur_stance.right_foot is not None:
            right_foot_task = ContactTask(
                self.robot, self.robot.right_foot, self.cur_stance.right_foot,
                weight=contact_weight)
        else:  # right_foot is swingign
            right_foot_task = LinkPoseTask(
                self.robot, self.robot.right_foot, self.swing_foot,
                weight=swing_weight)
        self.robot.ik.add_task(left_foot_task)
        self.robot.ik.add_task(right_foot_task)


class COMTube(object):

    """
    Primal tube of COM locations computed along with its dual acceleration cone.

    Parameters
    ----------
    start_com : array
        Start position of the COM.
    target_com : array
        End position of the COM.
    start_stance : Stance
        Stance used to compute the contact wrench cone.
    radius : scalar
        Side of the cross-section square (for ``shape`` > 2).
    margin : scalar
        Safety margin (in [m]) around boundary COM positions.

    Notes
    -----
    When there is an SS-to-DS contact switch, this strategy computes one primal
    tube and two dual intersection cones. The primal tube is a parallelepiped
    containing both the COM current and target locations. Its dual cone is used
    during the DS phase. The dual cone for the SS phase is calculated by
    intersecting the latter with the dual cone of the current COM position in
    single-contact.
    """

    def __init__(self, start_com, target_com, start_stance, next_stance, radius,
                 margin=0.01):
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


class COMTubePreviewControl(pymanoid.Process):

    """
    Feedback controller that continuously runs the preview controller and sends
    outputs to a COMAccelBuffer.

    Parameters
    ----------
    com : PointMass
        Current state (position and velocity) of the COM.
    fsm : WalkingFSM
        Instance of finite state machine.
    preview_buffer : PreviewBuffer
        MPC outputs are sent to this buffer.
    nb_mpc_steps : int
        Discretization step of the preview window.
    tube_radius : scalar
        Tube radius in [m] for the L1 norm.
    """

    def __init__(self, com, fsm, preview_buffer, nb_mpc_steps, tube_radius):
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

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
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
        # sim.log_comp_time('qp_solve', self.preview_control.solve_time)
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
            U = self.preview_control.U
            dT = [self.preview_control.timestep] * self.nb_mpc_steps
            self.preview_buffer.update_preview(U, dT, self.nb_mpc_steps)
            # <dirty why="used in PreviewDrawer">
            self.preview_buffer.switch_step = self.preview_control.switch_step
            # </dirty>
        except ValueError:
            print "MPC couldn't solve QP, constraints may be inconsistent"


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
        """
        Entry point called at each simulation tick.

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
        """
        if preview_buffer.is_empty:
            return
        com_pre, comd_pre = com_target.p, com_target.pd
        com_free, comd_free = com_target.p, com_target.pd
        self.handles = []
        self.handles.append(
            draw_point(com_target.p, color='m', pointsize=0.007))
        for preview_index in xrange(preview_buffer.nb_steps):
            com_pre0 = com_pre
            j = 3 * preview_index
            comdd = preview_buffer._U[j:j + 3]
            dT = preview_buffer._dT[preview_index]
            com_pre = com_pre + comd_pre * dT + comdd * .5 * dT ** 2
            comd_pre += comdd * dT
            color = \
                'b' if preview_index <= preview_buffer.switch_step \
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
        """
        Entry point called at each simulation tick.

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
        """
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

        Parameters
        ----------
        apex : array
            Position of the origin of the cone in world coordinates.
        axis : array
            Unit vector directing the cone axis and lying inside.
        combined : string, default='g-#'
            Drawing spec in matplotlib fashion.
        linewidth : scalar
            Thickness of the edges of the cone.
        pointsize : scalar
            Point size in [m].

        Returns
        -------
        handles : list of GUI handles
            Must be stored in some variable, otherwise the drawn object will
            vanish instantly.
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


class UpdateCOMTargetAccel(pymanoid.Process):

    def __init__(self, com_target, preview_buffer):
        super(UpdateCOMTargetAccel, self).__init__()
        self.com_target = com_target
        self.preview_buffer = preview_buffer

    def on_tick(self, sim):
        """
        Entry point called at each simulation tick.

        Parameters
        ----------
        sim : Simulation
            Instance of the current simulation.
        """
        self.com_target.pdd = self.preview_buffer.cur_control


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
        u_dim=3,
        callback=lambda u, dT: com_target.integrate_acceleration(u, dT))
    fsm = WalkingFSM(staircase, robot, swing_height=0.15, cycle=True)

    mpc = COMTubePreviewControl(
        com_target, fsm, preview_buffer,
        nb_mpc_steps=20,
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
    lf_traj_drawer = TrajectoryDrawer(robot.left_foot, 'g-')
    preview_drawer = PreviewDrawer()
    rf_traj_drawer = TrajectoryDrawer(robot.right_foot, 'r-')
    tube_drawer = TubeDrawer()
    update_com_target = UpdateCOMTargetAccel(com_target, preview_buffer)
    wrench_drawer = PointMassWrenchDrawer(com_target, fsm.get_cur_stance)

    sim.schedule_extra(com_traj_drawer)
    sim.schedule_extra(lf_traj_drawer)
    sim.schedule_extra(preview_drawer)
    sim.schedule_extra(rf_traj_drawer)
    sim.schedule_extra(tube_drawer)
    sim.schedule_extra(update_com_target)
    sim.schedule_extra(wrench_drawer)

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
