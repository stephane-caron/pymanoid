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

from numpy import array, dot

from body import Point
from contact_set import ContactSet
from misc import norm
from polyhedra import compute_polytope_hrep


class Stance(ContactSet):

    """
    Stances are contact sets with COM locations.

    Parameters
    ----------
    com : array or Point
        COM given by coordinates or a Point object.
    left_foot : Contact, optional
        Left foot contact.
    right_foot : Contact, optional
        Right foot contact.
    left_hand : Contact, optional
        Left hand contact.
    right_hand : Contact, optional
        Right hand contact.
    label : string, optional
        Label for the current contact phase.
    duration : double, optional
        Timing information.

    Note
    ----
    Stances proactively compute their contact wrench cone and static-equilibrium
    polygon when they are instanciated. The Stance class is therefore
    best-suited for planning; under time constraints, consider using bare
    ContactSets instead.
    """

    def __init__(self, com, left_foot=None, right_foot=None, left_hand=None,
                 right_hand=None, label=None, duration=None):
        contacts = filter(None, [left_foot, right_foot, left_hand, right_hand])
        super(Stance, self).__init__(contacts)
        if not issubclass(type(com), Point):
            com = Point(com, visible=False)
        self.com = com
        self.duration = duration
        self.label = label
        self.left_foot = left_foot
        self.left_hand = left_hand
        self.right_foot = right_foot
        self.right_hand = right_hand
        self.cwc = self.compute_wrench_face([0, 0, 0])  # calls cdd
        self.sep = Polytope(vertices=self.compute_static_equilibrium_polygon())
        self.sep.compute_hrep()
        A, _ = self.sep.hrep_pair
        self.sep_norm = array([norm(a) for a in A])

    @property
    def contact(self):
        """Unique contact if the Stance is single-support."""
        assert self.nb_contacts == 1
        for contact in self.contacts:
            return contact

    def dist_to_sep_edge(self, com):
        """
        Algebraic distance to the edge of the static-equilibrium polygon
        (positive inside, negative outside).
        """
        A, b = self.sep.hrep_pair
        alg_dists = (b - dot(A, com[:2])) / self.sep_norm
        return min(alg_dists)
