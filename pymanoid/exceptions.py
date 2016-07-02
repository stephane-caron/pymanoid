#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
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


from utils.exceptions import OptimalNotFound


class NaC(Exception):

    """Not A Cone"""

    def __init__(self, M):
        self.M = M

    def __str__(self):
        return "Matrix does not describe a polyhedral cone"


class NaP(Exception):

    """Not A Polytope"""

    def __init__(self, A, b):
        self.A = A
        self.b = b

    def __str__(self):
        return "(A * x <= b) does not describe a polytope"


class UnboundedPolyhedron(Exception):

    pass


class RobotNotFound(Exception):

    def __init__(self, robot_name):
        super(RobotNotFound, self).__init__("Robot '%s' not found" % robot_name)


__all__ = [
    'NaC',
    'OptimalNotFound',
    'UnboundedPolyhedron',
    'RobotNotFound'
]
