#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with pymanoid. If not, see <http://www.gnu.org/licenses/>.

from .body import Box
from .body import Cube
from .body import Manipulator
from .body import Point
from .body import PointMass
from .contact import Contact
from .contact import ContactFeed
from .contact import ContactSet
from .misc import error
from .misc import info
from .misc import warn
from .robot import Humanoid
from .robot import Robot
from .sim import Process
from .sim import Simulation
from .stance import Stance

import robots

__all__ = [
    'Box',
    'Contact',
    'ContactFeed',
    'ContactSet',
    'Cube',
    'Humanoid',
    'Humanoid',
    'Manipulator',
    'Point',
    'PointMass',
    'Process',
    'Robot',
    'Simulation',
    'Stance',
    'error',
    'info',
    'robots',
    'warn',
]

__version__ = '0.8'
