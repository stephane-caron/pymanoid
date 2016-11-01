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

import os
import sys

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/../')

from draw import draw_line
from generic import Drawer


class TrajectoryDrawer(Drawer):

    def __init__(self, body, combined='b-', color=None, linewidth=3,
                 linestyle=None):
        color = color if color is not None else combined[0]
        linestyle = linestyle if linestyle is not None else combined[1]
        assert linestyle in ['-', '.']
        self.body = body
        self.color = color
        self.handles = []
        self.last_pos = body.p
        self.linestyle = linestyle
        self.linewidth = linewidth

    def on_tick(self, sim):
        if self.linestyle == '-':
            self.handles.append(draw_line(
                self.last_pos, self.body.p, color=self.color,
                linewidth=self.linewidth))
        self.last_pos = self.body.p

    def dash_graph_handles(self):
        for i in xrange(len(self.handles)):
            if i % 2 == 0:
                self.handles[i] = None
