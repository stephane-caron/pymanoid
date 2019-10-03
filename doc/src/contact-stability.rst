*****************
Contact stability
*****************

Contact
=======

.. autoclass:: pymanoid.contact.Contact
    :members:

Multiple contacts
=================

.. autoclass:: pymanoid.contact.ContactSet
    :members:

Computing contact forces
========================

Contact wrenches exerted on the robot while it moves can be computed by
quadratic programming in the *wrench distributor* of a stance. This process is
automatically created when binding a :class:`Stance` to the robot model. It
will only be executed if you schedule it to your simulation. Here is a small
example:

.. code::

    from pymanoid import robots, Simulation, Stance

    sim = Simulation(dt=0.03)
    robot = robots.JVRC1(download_if_needed=True)
    stance = Stance(
        com=robot.get_com_point_mass(),
        left_foot=robot.left_foot.get_contact(pos=[0, 0.3, 0]),
        right_foot=robot.right_foot.get_contact(pos=[0, -0.3, 0]))
    stance.com.set_z(0.8)
    stance.bind(robot)
    sim.schedule(robot.ik)
    sim.schedule(robot.wrench_distributor)
    sim.start()

You can see the computed wrenches in the GUI by scheduling the corresponding
drawer process:

.. code::

    from pymanoid.gui import RobotWrenchDrawer

    sim.set_viewer()
    sim.schedule_extra(RobotWrenchDrawer(robot))

Once the wrench distributor is scheduled, it will store its outputs in the
contacts of the stance as well as in the robot's manipulators. Therefore, you
can access ``robot.left_foot.wrench`` or ``robot.stance.left_foot.wrench``
equivalently. Note that wrenches are given in the *world frame* rooted at their
respective contact points.

.. autoclass:: pymanoid.stance.StanceWrenchDistributor
   :members:
