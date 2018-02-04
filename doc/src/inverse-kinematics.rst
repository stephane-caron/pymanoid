******************
Inverse kinematics
******************

IK Solver
=========

.. autoclass:: pymanoid.ik.IKSolver
    :members:
    :undoc-members:

IK Tasks
========

.. automodule:: pymanoid.tasks
    :members:
    :undoc-members:

Example: Posture Generation
===========================

The example script ``examples/inverse_kinematics.py`` shows how to use the
robot IK to generate whole-body motions. Let us execute it step by step. First,
we initialize a simulation with a 30 ms timestep and load the JVRC-1 humanoid:

.. code::

    sim = pymanoid.Simulation(dt=0.03)
    robot = JVRC1('JVRC-1.dae', download_if_needed=True)
    sim.set_viewer()  # open GUI window
    sim.viewer.SetCamera([
        [-0.28985317,  0.40434422, -0.86746233,  2.73872042],
        [0.95680251,  0.10095043, -0.2726499,  0.86080128],
        [-0.02267371, -0.90901857, -0.41613837,  2.06654644],
        [0.,  0.,  0.,  1.]])

We define targets for foot contacts;

.. code::

    lf_target = Contact(robot.sole_shape, pos=[0, 0.3, 0], visible=True)
    rf_target = Contact(robot.sole_shape, pos=[0, -0.3, 0], visible=True)

Next, we set the altitude of the robot's free-flyer (attached to the waist
link) 80 cm above contacts:

.. code::

    robot.set_dof_values([0.8], dof_indices=[robot.TRANS_Z])

This being done, we initialize a point-mass that will serve as COM target for
the IK. Its initial position is set to ``robot.com``, which will be roughly 80
cm above contacts as it is close to the waist link:

.. code::

    com = PointMass(pos=robot.com, mass=robot.mass)

All our targets being defined, we initialize IK tasks for the feet and COM
position, as well as a posture task for the (necessary) regularization of the
underlying QP problem:

.. code::

    lf_task = ContactTask(robot, robot.left_foot, lf_target, weight=1.)
    rf_task = ContactTask(robot, robot.right_foot, rf_target, weight=1.)
    com_task = COMTask(robot, com, weight=1e-2)
    reg_task = PostureTask(robot, robot.q, weight=1e-6)

We initialize the robot's IK using all DOFs, and insert our tasks:

.. code::

    robot.init_ik(active_dofs=robot.whole_body)
    robot.ik.add_task(lf_task)
    robot.ik.add_task(rf_task)
    robot.ik.add_task(com_task)
    robot.ik.add_task(reg_task)

We can also throw in some extra DOF tasks for a nicer posture:

.. code::

    robot.ik.add_task(DOFTask(robot, robot.R_SHOULDER_R, -0.5, gain=0.5, weight=1e-5))
    robot.ik.add_task(DOFTask(robot, robot.L_SHOULDER_R, +0.5, gain=0.5, weight=1e-5))

Finally, call the IK solver to update the robot posture:

.. code::

    robot.ik.solve(max_it=100, impr_stop=1e-4)

The resulting posture looks like this:

.. figure:: images/inverse_kinematics.png
   :align:  center

   Robot posture found by inverse kinematics.
