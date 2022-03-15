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

from numpy import array
from os import system
from os.path import isfile

from ..body import Manipulator
from ..robot import Humanoid


class JVRC1(Humanoid):

    """
    JVRC-1 robot model.

    See <https://github.com/stephane-caron/openrave_models/tree/master/JVRC-1>.
    """

    MODEL_URL = 'https://raw.githubusercontent.com/stephane-caron/' \
        'openrave_models/master/JVRC-1/JVRC-1.dae'

    sole_shape = (0.11, 0.05)  # (half-length [m], half-width [m]) of foot sole
    leg_length = 0.85         # [m] (for a stretched leg)
    mass = 62.                # [kg]

    # DOF indices
    R_HIP_P = 0
    R_HIP_R = 1
    R_HIP_Y = 2
    R_KNEE_P = 3
    R_ANKLE_R = 4
    R_ANKLE_P = 5
    L_HIP_P = 6
    L_HIP_R = 7
    L_HIP_Y = 8
    L_KNEE_P = 9
    L_ANKLE_R = 10
    L_ANKLE_P = 11
    CHEST_Y = 12  # compatibility with HRP-4
    WAIST_Y = 12
    CHEST_P = 13  # compatibility with HRP-4
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
    ROT_Y = 47
    ROT_P = 48
    ROT_R = 49

    # Joints
    chest = [WAIST_Y, WAIST_P, WAIST_R]
    free_pos = [TRANS_X, TRANS_Y, TRANS_Z]
    free_rpy = [ROT_R, ROT_P, ROT_Y]
    left_ankle = [L_ANKLE_R, L_ANKLE_P]
    left_elbow = [L_ELBOW_P, L_ELBOW_Y]
    left_hip = [L_HIP_P, L_HIP_R, L_HIP_Y]
    left_index = [L_UINDEX, L_LINDEX]
    left_knee = [L_KNEE_P]
    left_little = [L_ULITTLE, L_LLITTLE]
    left_shoulder = [L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y]
    left_thumb = [L_UTHUMB, L_LTHUMB]
    left_wrist = [L_WRIST_R, L_WRIST_Y]
    neck = [NECK_Y, NECK_R, NECK_P]
    right_ankle = [R_ANKLE_R, R_ANKLE_P]
    right_elbow = [R_ELBOW_P, R_ELBOW_Y]
    right_hip = [R_HIP_P, R_HIP_R, R_HIP_Y]
    right_index = [R_UINDEX, R_LINDEX]
    right_knee = [R_KNEE_P]
    right_little = [R_ULITTLE, R_LLITTLE]
    right_shoulder = [R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y]
    right_thumb = [R_UTHUMB, R_LTHUMB]
    right_wrist = [R_WRIST_R, R_WRIST_Y]

    # Limbs
    left_arm = left_shoulder + left_elbow + left_wrist
    left_hand = left_thumb + left_index + left_little
    left_leg = left_hip + left_knee + left_ankle
    right_arm = right_shoulder + right_elbow + right_wrist
    right_hand = right_thumb + right_index + right_little
    right_leg = right_hip + right_knee + right_ankle

    # Body
    arms = left_arm + right_arm
    free = free_pos + free_rpy
    legs = left_leg + right_leg
    whole_body = arms + legs + chest + free

    # Half-sitting posture
    q_halfsit = array([
        -0.38, -0.01, 0., 0.72, -0.01, -0.33, -0.38, 0.02, 0., 0.72, -0.02,
        -0.33, 0., 0.13, 0., 0., 0., 0., -0.052, -0.17, 0., -0.52, 0., 0., 0.,
        0., 0., 0., 0., 0., 0., -0.052, 0.17, 0., -0.52, 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])

    # Compatibility
    chest_p_name = 'WAIST_P'
    chest_y_name = 'WAIST_Y'

    def __init__(self, path='JVRC-1.dae', root_body='PELVIS_S',
                 download_if_needed=True):
        """
        Add the JVRC-1 model to the environment.

        Parameters
        ----------
        path : string, optional
            Path to the COLLADA model of the robot.
        root_body : string, optional
            Name of the root body in the kinematic chain.
        download_if_needed : bool, optional
            If True and there is no model file in ``path``, will attempt to
            download it from ``JVRC1.MODEL_URL``.
        """
        if download_if_needed and not isfile(path):
            rc = system('wget %s -O %s' % (JVRC1.MODEL_URL, path))
            assert rc == 0, "Download of model file failed"
        super(JVRC1, self).__init__(path, root_body)
        self.ik.set_active_dofs(self.whole_body)
        self.left_foot = Manipulator(
            self.rave.GetManipulator("left_foot_base"), shape=self.sole_shape,
            friction=0.8)
        self.left_hand = Manipulator(
            self.rave.GetManipulator("left_hand_palm"), friction=0.8)
        self.mass = sum([link.GetMass() for link in self.rave.GetLinks()])
        self.right_foot = Manipulator(
            self.rave.GetManipulator("right_foot_base"), shape=self.sole_shape,
            friction=0.8)
        self.right_hand = Manipulator(
            self.rave.GetManipulator("right_hand_palm"), friction=0.8)
