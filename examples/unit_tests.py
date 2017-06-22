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

import unittest
import os
import sys

from numpy import dot, zeros

try:
    import pymanoid
except ImportError:
    script_path = os.path.realpath(__file__)
    sys.path.append(os.path.dirname(script_path) + '/../')
    import pymanoid

TINY = 1e-7


class TestContactSet(unittest.TestCase):
    """
    Test CWC computations.
    """

    def setUp(self):
        friction = 0.5
        shape = (0.10, 0.05)
        self.contact_set = pymanoid.ContactSet([
            pymanoid.Contact(
                shape=shape,
                pos=[0.20, 0.15, 0.1],
                rpy=[0.4, 0, 0],
                friction=friction),
            pymanoid.Contact(
                shape=shape,
                pos=[-0.2, -0.195, 0.],
                rpy=[-0.4, 0, 0],
                friction=friction)])

    def test_wrench_face_wrench_span(self):
        """
        Check that wrench rays are included in the H-representation.
        """
        p = zeros(3)
        F = self.contact_set.compute_wrench_face(p)
        S = self.contact_set.compute_wrench_span(p)
        for i in range(S.shape[1]):
            assert all(dot(F, S[:, i]) <= TINY)

    def test_supporting_wrench_from_span(self):
        """
        Compute supporting wrenches for wrench rays.
        """
        p = zeros(3)
        S = self.contact_set.compute_wrench_span(p)
        for i in range(S.shape[1]):
            s_i = S[:, i]
            res = self.contact_set.find_supporting_wrenches(
                s_i, p, solver='cvxopt')
            # see https://github.com/stephane-caron/pymanoid/issues/5
            # regarding the use of CVXOPT rather than quadprog here
            assert res is not None


if __name__ == "__main__":
    sim = pymanoid.Simulation(dt=0.03)
    unittest.main()
