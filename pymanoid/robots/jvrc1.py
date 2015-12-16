#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of pymanoid.
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


from pymanoid import Manipulator, Robot


class JVRC1(Robot):

    """
    Helper class for the JVRC-1 robot model.

    See https://github.com/stephane-caron/jvrc_models/tree/openrave/JVRC-1 for
    an OpenRAVE-compatible Collada model (JVRC-1.dae) and environment XML file.
    """

    mass = 62.  # [kg]

    R_HIP_P = 0
    R_HIP_R = 1
    R_HIP_Y = 2
    R_KNEE = 3
    R_ANKLE_R = 4
    R_ANKLE_P = 5
    L_HIP_P = 6
    L_HIP_R = 7
    L_HIP_Y = 8
    L_KNEE = 9
    L_ANKLE_R = 10
    L_ANKLE_P = 11
    WAIST_Y = 12
    WAIST_P = 13
    WAIST_R = 14
    NECK_Y = 15
    NECK_R = 16
    NECK_P = 17
    R_SHOULDER_P = 18
    R_SHOULDER_R = 19
    R_SHOULDER_Y = 20
    R_ELBOW_P = 21
    R_ELBOW_Y = 22
    R_WRIST_R = 23
    R_WRIST_Y = 24
    R_UTHUMB = 25
    R_LTHUMB = 26
    R_UINDEX = 27
    R_LINDEX = 28
    R_ULITTLE = 29
    R_LLITTLE = 30
    L_SHOULDER_P = 31
    L_SHOULDER_R = 32
    L_SHOULDER_Y = 33
    L_ELBOW_P = 34
    L_ELBOW_Y = 35
    L_WRIST_R = 36
    L_WRIST_Y = 37
    L_UTHUMB = 38
    L_LTHUMB = 39
    L_UINDEX = 40
    L_LINDEX = 41
    L_ULITTLE = 42
    L_LLITTLE = 43
    TRANS_X = 44
    TRANS_Y = 45
    TRANS_Z = 46
    ROT_R = 47
    ROT_P = 48
    ROT_Y = 49

    def __init__(self, env, robot_name='JVRC-1'):
        super(JVRC1, self).__init__(env, robot_name)
        rave = self.rave
        self.left_foot = Manipulator(rave.GetManipulator("left_foot_base"))
        self.right_foot = Manipulator(rave.GetManipulator("right_foot_base"))

    def get_dofs(self, *chains):
        l = []
        if 'right_leg' in chains:
            l.extend([
                self.R_HIP_R, self.R_HIP_Y, self.R_KNEE, self.R_ANKLE_R,
                self.R_ANKLE_P])
        if 'left_leg' in chains:
            l.extend([
                self.L_HIP_P, self.L_HIP_R, self.L_HIP_Y, self.L_KNEE,
                self.L_ANKLE_R, self.L_ANKLE_P])
        if 'chest' in chains:
            l.extend([
                self.WAIST_Y, self.WAIST_P, self.WAIST_R])
        if 'neck' in chains:
            l.extend([
                self.NECK_Y, self.NECK_R, self.NECK_P])
        if 'right_arm' in chains:
            l.extend([
                self.R_SHOULDER_P, self.R_SHOULDER_R, self.R_SHOULDER_Y,
                self.R_ELBOW_P, self.R_ELBOW_Y, self.R_WRIST_R, self.R_WRIST_Y])
        if 'right_hand' in chains:
            l.extend([
                self.R_UTHUMB, self.R_LTHUMB, self.R_UINDEX, self.R_LINDEX,
                self.R_ULITTLE, self.R_LLITTLE])
        if 'left_arm' in chains:
            l.extend([
                self.L_SHOULDER_P, self.L_SHOULDER_R, self.L_SHOULDER_Y,
                self.L_ELBOW_P, self.L_ELBOW_Y, self.L_WRIST_R, self.L_WRIST_Y])
        if 'left_hand' in chains:
            l.extend([
                self.L_UTHUMB, self.L_LTHUMB, self.L_UINDEX, self.L_LINDEX,
                self.L_ULITTLE, self.L_LLITTLE])
        if 'free' in chains:
            l.extend([
                self.TRANS_X, self.TRANS_Y, self.TRANS_Z,
                self.ROT_R, self.ROT_P, self.ROT_Y])
        return l
