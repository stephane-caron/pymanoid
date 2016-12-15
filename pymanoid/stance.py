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

from numpy import array, dot

from body import Point
from contact_set import ContactSet
from misc import norm
from polyhedra import Polytope


class Stance(ContactSet):

    """
    Stances extend contact sets with COM locations.
    """

    def __init__(self, com, left_foot=None, right_foot=None, left_hand=None,
                 right_hand=None, label=None, duration=None):
        """
        Create a new stance.

        INPUT:

        - ``com`` -- coordinates or Point object
        - ``left_foot`` -- (optional) left foot Contact
        - ``right_foot`` -- (optional) right foot Contact
        - ``left_hand`` -- (optional) left hand Contact
        - ``right_hand`` -- (optional) right hand Contact
        - ``label`` -- (optional) string label, e.g. phase in walking FSM
        - ``duration`` -- (optional) timing information
        """
        contacts = {}
        if type(com) is not Point:
            com = Point(com, visible=False)
        if left_foot:
            contacts['left_foot'] = left_foot
        if left_hand:
            contacts['left_hand'] = left_hand
        if right_foot:
            contacts['right_foot'] = right_foot
        if right_hand:
            contacts['right_hand'] = right_hand
        self.com = com
        self.duration = duration
        self.label = label
        self.left_foot = left_foot
        self.left_hand = left_hand
        self.right_foot = right_foot
        self.right_hand = right_hand
        super(Stance, self).__init__(contacts)
        self.compute_stability_criteria()

    def compute_stability_criteria(self):
        self.cwc = self.compute_wrench_face([0, 0, 0])  # calls cdd
        # self.cwc = compute_cwc_pyparma(self, [0, 0, 0])
        # m = RobotModel.mass  # however, the SEP does not depend on this
        self.sep = self.compute_static_equilibrium_polygon()
        self.sep_hrep = Polytope.hrep(self.sep)

    def dist_to_sep_edge(self, com):
        """
        Algebraic distance to the edge of the static-equilibrium polygon
        (positive inside, negative outside).
        """
        A, b = self.sep_hrep
        return min(b - dot(A, com[:2]))
