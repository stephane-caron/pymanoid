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


import numpy
import openravepy


__env__ = None


__gravity__ = numpy.array([0, 0, -9.80665])  # ISO 80000-3


def get_env():
    return __env__


def get_gravity():
    return __gravity__


def get_viewer():
    return __env__.GetViewer()


def register_env(env):
    global __env__
    __env__ = env


def init(env_file=None):
    env = openravepy.Environment()
    if env_file:
        env.Load(env_file)
    register_env(env)
    env.GetPhysicsEngine().SetGravity(__gravity__)
    env.SetViewer('qtcoin')
    viewer = env.GetViewer()
    viewer.SetBkgndColor([.6, .6, .8])
    viewer.SetCamera([  # default view from behind
        [0.,  0., 1., -3.],
        [-1., 0., 0.,  0.],
        [0., -1., 0.,  0.7],
        [0.,  0., 0.,  1.]])
