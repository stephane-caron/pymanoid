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
import time

from numpy import hstack, zeros

script_path = os.path.realpath(__file__)
sys.path.append(os.path.dirname(script_path) + '/../')

from draw import draw_force
from generic import Drawer


class ForceDrawer(Drawer):

    KO_COLOR = [.8, .4, .4]
    OK_COLOR = [1., 1., 1.]

    def __init__(self, com, contact_set, force_scale=0.0025):
        self.contact_set = contact_set
        self.force_scale = force_scale
        self.handles = []
        self.last_bkgnd_switch = None

    def on_tick(self, sim):
        """Find supporting contact forces at each COM acceleration update."""
        com = self.com
        comdd = com.pdd  # needs to be stored by the user
        gravity = sim.gravity
        wrench = hstack([com.mass * (comdd - gravity), zeros(3)])
        support = self.contact_set.find_supporting_forces(
            wrench, com.p, com.mass, 10.)
        if not support:
            self.handles = []
            sim.viewer.SetBkgndColor(self.KO_COLOR)
            self.last_bkgnd_switch = time.time()
        else:
            self.handles = [
                draw_force(c, fc, self.force_scale) for (c, fc) in support]
        if self.last_bkgnd_switch is not None \
                and time.time() - self.last_bkgnd_switch > 0.2:
            # let's keep epilepsy at bay
            sim.viewer.SetBkgndColor(self.OK_COLOR)
            self.last_bkgnd_switch = None
