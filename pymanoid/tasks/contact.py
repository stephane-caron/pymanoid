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

from numpy import array, dot

from generic import Task
from link import LinkPoseTask


_oppose_quat = array([-1., -1., -1., -1., +1., +1., +1.])


class ContactTask(LinkPoseTask):

    task_type = 'contact'

    def __init__(self, robot, link, target, **kwargs):
        if type(target) is list:
            target = array(target)

        def pos_residual():
            residual = target.contact_pose - link.pose
            if dot(residual[0:4], residual[0:4]) > 1.:
                return _oppose_quat * target.contact_pose - link.pose
            return residual

        def jacobian():
            return robot.compute_link_pose_jacobian(link)

        self.link = link  # used by LinkPoseTask.name
        Task.__init__(self, jacobian, pos_residual=pos_residual, **kwargs)
