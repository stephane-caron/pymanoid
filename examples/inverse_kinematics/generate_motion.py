#!/usr/bin/env python
# -*- coding: utf-8 -*-

import openravepy
import pylab
import pymanoid
import time

from pymanoid.robots import JVRC1
from numpy import arange, array, sin


dt = 1e-2  # [s]

# gains
G_com = 100.
G_link = 100.
G_dof = 0.5
G_ref = 0.5
qd_lim = 1.  # [rad/s]

# weights in objective function
w_lnk = 100.
w_com = 001.
w_reg = 001.
w_dof = 000.1
w_ref = 000.1


if __name__ == '__main__':
    pylab.ion()
    env = openravepy.Environment()
    env.Load('JVRC-1.xml')
    env.SetViewer('qtcoin')
    robot = JVRC1(env)

    viewer = env.GetViewer()
    viewer.SetBkgndColor([.7, .7, .9])
    viewer.SetCamera(array([
        [-0.28985317,  0.40434422, -0.86746233,  2.73872042],
        [0.95680251,  0.10095043, -0.2726499,  0.86080128],
        [-0.02267371, -0.90901857, -0.41613837,  2.06654644],
        [0.,  0.,  0.,  1.]]))

    # Initial robot pose
    robot.set_transparency(0.4)
    robot.scale_dof_limits(0.95)
    dof_objectives = [  # will also be passed to the IK
        (robot.R_SHOULDER_R, -.5),
        (robot.L_SHOULDER_R, +.5)]
    q_init = robot.q.copy()
    for (dof_id, dof_ref) in dof_objectives:
        robot.set_dof_values([dof_ref], [dof_id])
        q_init[dof_id] = dof_ref
    robot.set_dof_values([-1], [robot.R_SHOULDER_P])
    robot.set_dof_values([-1], [robot.L_SHOULDER_P])
    robot.set_dof_values([0.8], dof_indices=[robot.TRANS_Z])

    # set active DOFs for the IK
    active_dofs = robot.get_dofs(
        'chest', 'free', 'left_arm', 'right_arm', 'left_leg', 'right_leg')
    robot.set_active_dofs(active_dofs)

    # IK targets: COM and foot poses
    com = pymanoid.Box(env, box_dim=0.05, pos=robot.com, color='g')
    init_com = com.p.copy()
    left_foot_target = pymanoid.Box(
        env, X=0.224 / 2, Y=0.130 / 2, Z=0.01, pos=[0, 0.3, 0])
    right_foot_target = pymanoid.Box(
        env, X=0.224 / 2, Y=0.130 / 2, Z=0.01, pos=[0, -0.3, 0])

    # Initialize the IK
    robot.init_ik(qd_lim)
    robot.add_link_objective(
        robot.left_foot, left_foot_target, G_link, w_lnk)
    robot.add_link_objective(
        robot.right_foot, right_foot_target, G_link, w_lnk)
    robot.add_com_objective(com, G_com, w_com)
    robot.add_posture_objective(robot.q, G_ref, w_ref)
    for (dof_id, dof_ref) in dof_objectives:
        robot.add_dof_objective(dof_id, dof_ref, G_dof, w_dof)
    robot.add_velocity_regularization(w_reg)

    print ""
    print "First, we will solve for an initial posture, enforcing foot contacts"
    print "while keeping the center of mass (COM) at its current position."
    print ""
    raw_input("Ready to call solve_ik()? ")

    robot.solve_ik(dt, max_it=100, conv_tol=1e-4, debug=True)

    print ""
    print "Now, we will move the target COM (green box) back and forth, and"
    print "have the robot follow it using the step_tracker() function."
    print ""
    raw_input("Ready to go for 10 seconds? ")

    for t in arange(0, 10, 1e-2):
        loop_start = time.time()
        com_var = sin(t) * array([.2, 0, 0])
        com.set_pos(init_com + array([-0.2, 0., 0.]) + com_var)
        robot.step_tracker(dt)
        rem_time = dt - (time.time() - loop_start)
        if rem_time > 0:
            time.sleep(rem_time)

    print ""
    print "This example is over. You can now play with the"
    print "`robot` object in the following Python shell."
    print ""
    import IPython
    IPython.embed()
