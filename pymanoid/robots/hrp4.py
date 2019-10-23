#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2019 Stephane Caron <stephane.caron@lirmm.fr>
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

from numpy import array, pi

from ..body import Manipulator
from ..robot import Humanoid
from ..tasks import DOFTask, MinCAMTask


class HRP4(Humanoid):

    """
    HRP-4 (HRP4R) humanoid robot.

    This file includes information that is publicly released in
    <https://doi.org/10.1109/IROS.2011.6094465> or over the Web for the total
    mass and kinematic chain. Other information is read from the COLLADA model.
    """

    leg_length = 0.8           # [m]   (roughly, for a strechted leg)
    mass = 39.                 # [kg]  (includes batteries)
    sole_shape = (0.11, 0.05)  # (half-length [m], half-width [m]) of foot sole
    palm_shape = (0.03, 0.04)  # (half-length [m], half-width [m]) of palm surf

    # DOF indexes with respect to COLLADA model
    L_F53 = 0
    L_F52 = 1
    L_F43 = 2
    L_F42 = 3
    L_F33 = 4
    L_F32 = 5
    L_F23 = 6
    L_F22 = 7
    R_F53 = 8
    R_F52 = 9
    R_F43 = 10
    R_F42 = 11
    R_F33 = 12
    R_F32 = 13
    R_F23 = 14
    R_F22 = 15
    R_HIP_Y = 16
    R_HIP_R = 17
    R_HIP_P = 18
    R_KNEE_P = 19
    R_ANKLE_P = 20
    R_ANKLE_R = 21
    L_HIP_Y = 22
    L_HIP_R = 23
    L_HIP_P = 24
    L_KNEE_P = 25
    L_ANKLE_P = 26
    L_ANKLE_R = 27
    CHEST_P = 28
    WAIST_P = 28  # compatibility with JVRC-1
    CHEST_Y = 29
    WAIST_Y = 29  # compatibility with JVRC-1
    NECK_Y = 30
    NECK_P = 31
    R_SHOULDER_P = 32
    R_SHOULDER_R = 33
    R_SHOULDER_Y = 34
    R_ELBOW_P = 35
    R_WRIST_Y = 36
    R_WRIST_P = 37
    R_WRIST_R = 38
    R_HAND_J0 = 39
    R_HAND_J1 = 40
    L_SHOULDER_P = 41
    L_SHOULDER_R = 42
    L_SHOULDER_Y = 43
    L_ELBOW_P = 44
    L_WRIST_Y = 45
    L_WRIST_P = 46
    L_WRIST_R = 47
    L_HAND_J0 = 48
    L_HAND_J1 = 49
    TRANS_X = 50
    TRANS_Y = 51
    TRANS_Z = 52
    ROT_Y = 53
    ROT_P = 54
    ROT_R = 55

    # Joints
    chest = [CHEST_P, CHEST_Y]
    free_pos = [TRANS_X, TRANS_Y, TRANS_Z]
    free_rpy = [ROT_R, ROT_P, ROT_Y]
    left_ankle = [L_ANKLE_P, L_ANKLE_R]
    left_elbow = [L_ELBOW_P]
    left_hip = [L_HIP_Y, L_HIP_R, L_HIP_P]
    left_knee = [L_KNEE_P]
    left_shoulder = [L_SHOULDER_P, L_SHOULDER_R, L_SHOULDER_Y]
    left_thumb = [L_HAND_J0, L_HAND_J1]
    left_wrist = [L_WRIST_Y, L_WRIST_P, L_WRIST_R]
    neck = [NECK_Y, NECK_P]
    right_ankle = [R_ANKLE_P, R_ANKLE_R]
    right_elbow = [R_ELBOW_P]
    right_hip = [R_HIP_Y, R_HIP_R, R_HIP_P]
    right_knee = [R_KNEE_P]
    right_shoulder = [R_SHOULDER_P, R_SHOULDER_R, R_SHOULDER_Y]
    right_thumb = [R_HAND_J0, R_HAND_J1]
    right_wrist = [R_WRIST_Y, R_WRIST_P, R_WRIST_R]

    # Limbs
    left_arm = left_shoulder + left_elbow + left_wrist
    left_leg = left_hip + left_knee + left_ankle
    right_arm = right_shoulder + right_elbow + right_wrist
    right_leg = right_hip + right_knee + right_ankle

    # Body
    arms = left_arm + right_arm
    free = free_pos + free_rpy
    legs = left_leg + right_leg
    upper_body = arms + chest
    whole_body = arms + legs + chest + free

    # Half-sitting posture
    q_halfsit = pi / 180 * array(
        [0., 0., 0., 0., 0., 0., 0., 0., 0.,  0., 0., 0., 0., 0., 0.,
         0., 0., -0.76, -22.02, 41.29, -18.75, -0.45, 0., 1.15, -21.89,
         41.21, -18.74, -1.10, 8., 0., 0., 0., -3., -10., 0., -30., 0.,
         0., 0., 0., 0., -3., 10., 0., -30., 0., 0., 0., 0.,  0., 0., 0., 0.,
         0., 0., 0.])

    # Compatibility
    chest_p_name = 'CHEST_P'
    chest_y_name = 'CHEST_Y'

    def __init__(self, path='HRP4R.dae', root_body='BODY'):
        """
        Add the HRP4R model to the environment.

        Parameters
        ----------
        path : string
            Path to the COLLADA model of the robot.
        root_body : string
            Name of the root body in the kinematic chain.

        Note
        ----
        Unfortunately it is unclear whether we can release the COLLADA file
        ``HRP4R.dae`` (md5sum: 38c30928b7ae62daa0fc67ed0488b0a1) due to
        copyright.
        """
        super(HRP4, self).__init__(path, root_body)
        self.ik.set_active_dofs(self.whole_body)
        self.mass = sum([link.GetMass() for link in self.rave.GetLinks()])
        self.left_foot = Manipulator(
            self.rave.GetManipulator("LeftFootCenter"), shape=self.sole_shape,
            friction=0.8)
        self.left_hand = Manipulator(
            self.rave.GetManipulator("LeftHandCenter"), shape=self.palm_shape,
            friction=0.8)
        self.right_foot = Manipulator(
            self.rave.GetManipulator("RightFootCenter"), shape=self.sole_shape,
            friction=0.8)
        self.right_hand = Manipulator(
            self.rave.GetManipulator("RightHandCenter"), shape=self.palm_shape,
            friction=0.8)

    def add_shoulder_abduction_task(self, weight=None):
        self.ik.add(DOFTask(self, self.R_SHOULDER_R, -0.4, weight))
        self.ik.add(DOFTask(self, self.L_SHOULDER_R, +0.4, weight))

    def add_shoulder_extension_task(self, weight=None):
        self.ik.add(DOFTask(self, self.L_SHOULDER_P, +0.5, weight))
        self.ik.add(DOFTask(self, self.R_SHOULDER_P, +0.5, weight))

    def add_shoulder_flexion_task(self, weight=None):
        self.ik.add(DOFTask(self, self.L_SHOULDER_P, -0.5, weight))
        self.ik.add(DOFTask(self, self.R_SHOULDER_P, -0.5, weight))

    def add_shoulder_neutral_pitch_task(self, weight=None):
        self.ik.add(DOFTask(self, self.L_SHOULDER_P, 0., weight))
        self.ik.add(DOFTask(self, self.R_SHOULDER_P, 0., weight))

    def add_upright_chest_task(self, weight=None):
        self.ik.add(DOFTask(self, self.ROT_P, 0., weight))
        self.ik.add(DOFTask(self, self.CHEST_P, 0.2, weight))
        self.ik.add(DOFTask(self, self.CHEST_Y, 0., weight))

    def setup_ik_for_walking(self, com_target):
        """
        Prepare inverse-kinematic tracking for locomotion.

        Parameters
        ----------
        com_target : pymanoid.PointMass
            Target for the IK task on the center of mass.
        """
        self.ik.tasks['COM'].update_target(com_target)
        self.ik.add(MinCAMTask(self))
        self.add_upright_chest_task()
        self.add_shoulder_neutral_pitch_task()
        self.ik.set_weights({
            self.left_foot.name:  1.,
            self.right_foot.name: 1.,
            'COM': 1e-2,
            'MIN_CAM': 1e-4,
            'ROT_P': 1e-4,
            'CHEST_P': 1e-4,
            'CHEST_Y': 1e-4,
            'L_SHOULDER_P': 1e-5,
            'R_SHOULDER_P': 1e-5,
        })

    def suntan(self, amount=0.3):
        """
        Prepare model for screenshots on white background ;)

        Parameters
        ----------
        amount : scalar
            Number between 0. (no suntan) and 1. (full burn).
        """
        ambient, diffuse = 0., 1. - amount
        for link in self.rave.GetLinks():
            if len(link.GetGeometries()) > 0:
                geom = link.GetGeometries()[0]
                geom.SetAmbientColor([ambient] * 3)
                geom.SetDiffuseColor([diffuse] * 3)
