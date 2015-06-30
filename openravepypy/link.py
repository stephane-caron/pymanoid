#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of openravepypy.
#
# openravepypy is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# openravepypy is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openravepypy. If not, see <http://www.gnu.org/licenses/>.


from numpy import dot


class Link(object):

    def __init__(self, rave_link):
        self.rave = rave_link

    @property
    def index(self):
        return self.rave.GetIndex()

    @property
    def name(self):
        return self.rave.GetName()

    @property
    def T(self):
        return self.rave.GetTransform()

    @property
    def R(self):
        return self.T[0:3, 0:3]

    @property
    def p(self):
        return self.T[0:3, 3]

    @property
    def pos(self):
        return self.p

    @property
    def pose(self):
        pose = self.rave.GetTransformPose()
        if pose[0] < 0:  # convention: cos(alpha) > 0
            # this convention enforces Slerp shortest path
            pose[:4] *= -1
        return pose


class LinkPoint(Link):

    def __init__(self, rave_link, p_local):
        super(LinkPoint, self).__init__(rave_link)
        self.p_local = p_local

    @property
    def p(self):
        p0 = self.T[0:3, 3]
        return p0 + dot(self.R, self.p_local)

    @property
    def pos(self):
        return self.p

    @property
    def pose(self):
        pose = self.rave.GetTransformPose()
        R = self.rave.GetTransform()[0:3, 0:3]
        pose[4:] += dot(R, self.p_local)
        if pose[0] < 0:  # convention: cos(alpha) > 0
            # this convention enforces Slerp shortest path
            pose[:4] *= -1
        return pose
