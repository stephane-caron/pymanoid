#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2020 Stephane Caron <stephane.caron@normalesup.org>
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

from .body import Body
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
from .models import InvertedPendulum
from .proc import CameraRecorder
from .proc import JointRecorder
from .proc import Process
from .robot import Humanoid
from .robot import Robot
from .sim import Simulation
from .stance import Stance
from .swing_foot import SwingFoot

import models
import robots

__all__ = [
    'Body',
    'Box',
    'CameraRecorder',
    'Contact',
    'ContactFeed',
    'ContactSet',
    'Cube',
    'Humanoid',
    'Humanoid',
    'InvertedPendulum',
    'Manipulator',
    'Point',
    'PointMass',
    'Process',
    'Robot',
    'Simulation',
    'Stance',
    'SwingFoot',
    'error',
    'info',
    'models',
    'robots',
    'warn',
]

__version__ = '1.2.0'
