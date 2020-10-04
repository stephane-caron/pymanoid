******************
Inverse kinematics
******************

Inverse kinematics (IK) is the problem of computing *motions* (velocities,
accelerations) that make the robot achieve a given set of *tasks*, such as
putting a foot on a surface, moving the center of mass (COM) to a target
location, etc. If you are not familiar with these concepts, check out `this
introduction to inverse kinematics
<https://scaron.info/teaching/inverse-kinematics.html>`_.

Tasks
=====

Tasks are the way to specify objectives to the robot model in a human-readable
way.

.. automodule:: pymanoid.tasks
    :members:

Solver
======

The IK solver is the numerical optimization program that converts task targets
and the current robot state to joint motions. In pymanoid, joint motions are
computed as velocities that are integrated forward during each simulation cycle
(other IK solvers may compute acceleration or jerk values, which are then
integrated twice or thrice respectively).

.. autoclass:: pymanoid.ik.IKSolver
    :members:

Acceleration limits
-------------------

When the robot model has joint acceleration limits, *i.e.* when a vector of
size ``robot.nb_dofs`` is specified in ``robot.qdd_lim`` at initialization, the
inverse kinematics will include them in its optimization problem. The
formulation is more complex than a finite-difference approximation: joint
velocities will be selected so that (1) the joint does not collide with its
position limit in one iteration, but also (2) despite its acceleration limit,
it can still brake fast enough to avoid colliding with its position limit in
the future. The inverse kinematics in pymanoid implements the solution to this
problem described in Equation (14) of [Flacco15]_.

Stance
======

The most common IK tasks for humanoid locomotion are the `COM task
<#pymanoid.tasks.COMTask>`_ and `end-effector pose task
<#pymanoid.tasks.PoseTask>`_. The ``Stance`` class avoids the hassle of
creating and adding these tasks one by one. To use it, first create your targets:

.. code::

    com_target = robot.get_com_point_mass()
    lf_target = robot.left_foot.get_contact(pos=[0, 0.3, 0])
    rf_target = robot.right_foot.get_contact(pos=[0, -0.3, 0])

Then create the stance and bind it to the robot:

.. code::

   stance = Stance(com=com_target, left_foot=lf_target, right_foot=rf_target)
   stance.bind(robot)

Calling the `bind <#pymanoid.stance.Stance.bind>`_ function will automatically add the corresponding tasks to the robot IK solver. See the `inverse_kinematics.py <https://github.com/stephane-caron/pymanoid/blob/master/examples/inverse_kinematics.py>`_ example for more details.

.. autoclass:: pymanoid.stance.Stance
    :members:

Example
=======

In this example, we will see how to put the humanoid robot model in a desired
configuration. Let us initialize a simulation with a 30 ms timestep and load
the JVRC-1 humanoid:

.. code::

    sim = pymanoid.Simulation(dt=0.03)
    robot = JVRC1('JVRC-1.dae', download_if_needed=True)
    sim.set_viewer()  # open GUI window

We define targets for foot contacts;

.. code::

    lf_target = Contact(robot.sole_shape, pos=[0, 0.3, 0], visible=True)
    rf_target = Contact(robot.sole_shape, pos=[0, -0.3, 0], visible=True)

Next, let us set the altitude of the robot's free-flyer (attached to the waist
link) 80 cm above contacts. This is useful to give the IK solver a good initial
guess and avoid coming out with a valid but weird solution in the end.

.. code::

    robot.set_dof_values([0.8], dof_indices=[robot.TRANS_Z])

This being done, we initialize a point-mass that will serve as center-of-mass
(COM) target for the IK. Its initial position is set to ``robot.com``, which
will be roughly 80 cm above contacts as it is close to the waist link:

.. code::

    com_target = PointMass(pos=robot.com, mass=robot.mass)

All our targets being defined, we initialize IK tasks for the feet and COM
position, as well as a posture task for the (necessary) regularization of the
underlying QP problem:

.. code::

    lf_task = ContactTask(robot, robot.left_foot, lf_target, weight=1.)
    rf_task = ContactTask(robot, robot.right_foot, rf_target, weight=1.)
    com_task = COMTask(robot, com_target, weight=1e-2)
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

Alternatively, rather than creating and adding all tasks one by one, we could
have used the `Stance <#pymanoid.stance.Stance>`_ interface:

.. code::

    stance = Stance(com=com_target, left_foot=lf_target, right_foot=rf_target)
    stance.bind(robot)
    robot.ik.add_task(DOFTask(robot, robot.R_SHOULDER_R, -0.5, gain=0.5, weight=1e-5))
    robot.ik.add_task(DOFTask(robot, robot.L_SHOULDER_R, +0.5, gain=0.5, weight=1e-5))
    robot.ik.solve(max_it=100, impr_stop=1e-4)

This code is more concise and yields the same result.
