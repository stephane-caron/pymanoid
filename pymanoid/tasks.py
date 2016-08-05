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

from tasks_contact import ContactTask
from tasks_contact import LinkPosTask
from tasks_contact import LinkPoseTask
from tasks_dof import DOFTask
from tasks_dof import MinAccelerationTask
from tasks_dof import MinVelocityTask
from tasks_dof import PostureTask
from tasks_centroid import COMTask
from tasks_centroid import ConstantCAMTask
from tasks_centroid import MinCAMTask

__all__ = [
    'COMTask',
    'ConstantCAMTask',
    'ContactTask',
    'DOFTask',
    'LinkPosTask',
    'LinkPoseTask',
    'MinAccelerationTask',
    'MinCAMTask',
    'MinVelocityTask',
    'PostureTask',
]
