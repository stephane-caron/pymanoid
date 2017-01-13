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

from numpy import hstack

from body import Box
from misc import interpolate_pose_linear
from process import Process
from rotations import quat_slerp, rotation_matrix_from_quat
from tasks import ContactTask, LinkPoseTask


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

        INPUT:

        - ``start_pose`` -- new start pose
        - ``end_pose`` -- new end pose
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

        INPUT:

        - ``s`` -- index between 0 and 1 in the swing-foot motion.
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


class WalkingFSM(Process):

    """
    Finite State Machine for biped walking.
    """

    def __init__(self, stances, robot, swing_foot, cycle=False):
        """
        Create a new finite state machine.

        INPUT:

        - ``stances`` -- list of Stance objects
        - ``robot`` -- Robot object
        - ``swing_foot`` -- SwingFoot object
        - ``cycle`` -- (optional) first stance follows the last one
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
        self.swing_foot = swing_foot
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

        INPUT:

        - ``sim`` -- instance of current simulation
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
