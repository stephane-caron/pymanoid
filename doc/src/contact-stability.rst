*****************
Contact stability
*****************

Single contact
==============

.. autoclass:: pymanoid.contact.Contact
    :members:

Multi-contact
=============

.. autoclass:: pymanoid.contact.ContactSet
    :members:

Contact wrenches
================

Contact wrenches exerted on the robot while it moves can be computed by
quadratic programming in the *wrench distributor* of a stance. This process is
automatically created when binding a :class:`Stance` to the robot model. It
will only be executed if you schedule it to your simulation:

.. code::

    stance.bind(robot)
    sim.schedule(robot.wrench_distributor)

Once the wrench distributor is scheduled, it will store its outputs in the
contacts of the stance as well as in the robot's manipulators. Therefore, you
can access ``robot.left_foot.wrench`` or ``robot.stance.left_foot.wrench``
equivalently. Note that wrenches are given in their respective contact frames,
not in the world frame.

.. autoclass:: pymanoid.stance.StanceWrenchDistributor
   :members:
