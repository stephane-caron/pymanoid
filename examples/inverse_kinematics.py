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


import IPython
import numpy
import pymanoid
import time


if __name__ == '__main__':
    pymanoid.init()
    robot = pymanoid.robots.JVRC1('JVRC-1.dae', download_if_needed=True)

    viewer = pymanoid.get_env().GetViewer()
    viewer.SetCamera([
        [-0.28985317,  0.40434422, -0.86746233,  2.73872042],
        [0.95680251,  0.10095043, -0.2726499,  0.86080128],
        [-0.02267371, -0.90901857, -0.41613837,  2.06654644],
        [0.,  0.,  0.,  1.]])

    # Initial robot pose
    robot.set_transparency(0.4)
    robot.scale_dof_limits(0.95)
    dof_targets = [  # will also be passed to the IK
        (robot.R_SHOULDER_R, -.5),
        (robot.L_SHOULDER_R, +.5)]
    q_init = robot.q.copy()
    for (dof_id, dof_ref) in dof_targets:
        robot.set_dof_values([dof_ref], [dof_id])
        q_init[dof_id] = dof_ref
    robot.set_dof_values([-1], [robot.R_SHOULDER_P])
    robot.set_dof_values([-1], [robot.L_SHOULDER_P])
    robot.set_dof_values([0.8], dof_indices=[robot.TRANS_Z])

    # set active DOFs for the IK
    active_dofs = robot.chest + robot.free + robot.left_arm + \
        robot.right_arm + robot.left_leg + robot.right_leg
    robot.set_active_dofs(active_dofs)

    # IK targets: COM and foot poses
    com = pymanoid.Cube(0.05, pos=robot.com, color='g')
    init_com = com.p.copy()
    left_foot_target = pymanoid.Contact(
        X=0.224 / 2,
        Y=0.130 / 2,
        Z=0.01,
        pos=[0, 0.3, 0],
        visible=True)
    right_foot_target = pymanoid.Contact(
        X=0.224 / 2,
        Y=0.130 / 2,
        Z=0.01,
        pos=[0, -0.3, 0],
        visible=True)

    # Initialize the IK
    robot.init_ik(gains=None, weights=None)  # using default IK settings
    robot.add_contact_task(robot.left_foot, left_foot_target)
    robot.add_contact_task(robot.right_foot, right_foot_target)
    robot.add_com_task(com)
    robot.add_posture_task(robot.q)
    for (dof_id, dof_ref) in dof_targets:
        robot.add_dof_task(dof_id, dof_ref, gain=0.5, weight=0.1)

    print ""
    print "First, we solve for an initial posture, enforcing foot contacts"
    print "while keeping the center of mass (COM) at its current position."
    print "The numbers below show the objective value at each iteration:"
    print ""
    robot.solve_ik(max_it=100, conv_tol=1e-4, debug=True)

    print ""
    print "Now, we move the target COM (green box) back and forth, and"
    print "have the robot follow it using the step_ik() function."
    print ""

    dt = 1e-2  # [s]
    for t in numpy.arange(0, 10, 1e-2):
        loop_start = time.time()
        com_var = numpy.sin(t) * numpy.array([.2, 0, 0])
        com.set_pos(init_com + numpy.array([-0.2, 0., 0.]) + com_var)
        robot.step_ik(dt)
        rem_time = dt - (time.time() - loop_start)
        if rem_time > 0:
            time.sleep(rem_time)

    print "Finally, we launch the IK thread. Try moving the green box!"
    print ""
    robot.start_ik_thread(dt)
    IPython.embed()
