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

from body import Body
from body import Box
from body import Cube
from body import Link
from body import Manipulator
from body import PointMass
from contact import Contact
from contact import ContactSet
from draw import draw_2d_cone
from draw import draw_3d_cone
from draw import draw_arrow
from draw import draw_force
from draw import draw_line
from draw import draw_point
from draw import draw_points
from draw import draw_polygon
from draw import draw_polyhedron
from env import get_env
from env import get_gravity
from env import get_viewer
from env import init
from env import read_window_id
from env import register_env
from env import set_camera_back
from env import set_camera_bottom
from env import set_camera_front
from env import set_camera_left
from env import set_camera_right
from env import set_camera_top
from env import set_viewer
from env import take_screenshot
from humanoid import Humanoid
from optim import solve_lp
from optim import solve_qp
from robot import Robot

import robots

__all__ = [
    'Body',
    'Box',
    'Contact',
    'ContactSet',
    'Cube',
    'Humanoid',
    'Link',
    'Manipulator',
    'PointMass',
    'Robot',
    'draw_2d_cone',
    'draw_3d_cone',
    'draw_arrow',
    'draw_force',
    'draw_line',
    'draw_point',
    'draw_points',
    'draw_polygon',
    'draw_polyhedron',
    'get_env',
    'get_gravity',
    'get_viewer',
    'init',
    'read_window_id',
    'register_env',
    'robots',
    'set_camera_back',
    'set_camera_bottom',
    'set_camera_front',
    'set_camera_left',
    'set_camera_right',
    'set_camera_top',
    'set_viewer',
    'solve_lp',
    'solve_qp',
    'take_screenshot',
]
