#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
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


from env import get_viewer


def set_camera_above(x=0, y=0, z=3):
    get_viewer().SetCamera([
        [0, -1,  0, x],
        [-1, 0,  0, y],
        [0,  0, -1, z],
        [0,  0,  0, 1.]])


def set_camera_behind(x=-3, y=0, z=0.7):
    get_viewer().SetCamera([
        [0,  0, 1, x],
        [-1, 0, 0, y],
        [0, -1, 0, z],
        [0,  0, 0, 1.]])


def set_camera_below(x=0, y=0, z=-2):
    get_viewer().SetCamera([
        [0, -1, 0, x],
        [1,  0, 0, y],
        [0,  0, 1, z],
        [0,  0, 0, 1]])


def set_camera_front(x=+3, y=0, z=0.7):
    get_viewer().SetCamera([
        [0,  0, -1, x],
        [1,  0,  0, y],
        [0, -1,  0, z],
        [0,  0,  0, 1.]])


def set_camera_left(x=0, y=+3, z=0.7):
    get_viewer().SetCamera([
        [-1, 0,  0, x],
        [0,  0, -1, y],
        [0, -1,  0, z],
        [0,  0,  0, 1.]])


def set_camera_right(x=0, y=-3, z=0.7):
    get_viewer().SetCamera([
        [1,  0,  0, x],
        [0,  0, 1, y],
        [0, -1, 0, z],
        [0,  0, 0, 1.]])
