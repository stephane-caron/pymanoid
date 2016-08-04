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

from robot_centroid import CentroidalRobot
from contact import ContactSet


class Humanoid(CentroidalRobot):

    """
    Posture generation
    ==================
    """

    def generate_posture_from_contacts(self, contact_set, com_target=None,
                                       *args, **kwargs):
        assert self.ik is not None, \
            "Initialize the IK before generating posture"
        if 'left_foot' in contact_set:
            self.add_contact_task(self.left_foot, contact_set['left_foot'])
        if 'right_foot' in contact_set:
            self.add_contact_task(self.right_foot, contact_set['right_foot'])
        if 'left_hand' in contact_set:
            self.add_contact_task(self.left_hand, contact_set['left_hand'])
        if 'right_hand' in contact_set:
            self.add_contact_task(self.right_hand, contact_set['right_hand'])
        if com_target is not None:
            self.add_com_task(com_target)
        self.add_posture_task(self.q_halfsit)
        # warm start on halfsit posture
        # self.set_dof_values(self.q_halfsit)
        # => no, user may want to override this
        self.solve_ik(*args, **kwargs)

    def generate_posture(self, stance, com_target=None, *args, **kwargs):
        assert self.ik is not None, \
            "Initialize the IK before generating posture"
        if type(stance) is ContactSet:
            return self.generate_posture_from_contacts(stance, com_target)
        if hasattr(stance, 'com'):
            self.add_com_task(stance.com)
        elif hasattr(stance, 'com_target'):
            self.add_com_task(stance.com_target)
        elif com_target is not None:
            self.add_com_task(com_target)
        if hasattr(stance, 'left_foot'):
            self.add_contact_task(
                self.left_foot, stance.left_foot)
        if hasattr(stance, 'right_foot'):
            self.add_contact_task(
                self.right_foot, stance.right_foot)
        if hasattr(stance, 'left_hand'):
            self.add_contact_task(
                self.left_hand, stance.left_hand)
        if hasattr(stance, 'right_hand'):
            self.add_contact_task(
                self.right_hand, stance.right_hand)
        self.add_posture_task(self.q_halfsit)
        # warm start on halfsit posture
        # self.set_dof_values(self.q_halfsit)
        # => no, user may want to override this
        self.solve_ik(*args, **kwargs)
