************************
Graphical user interface
************************

Primitive functions
===================

.. autofunction:: pymanoid.gui.draw_arrow
.. autofunction:: pymanoid.gui.draw_force
.. autofunction:: pymanoid.gui.draw_line
.. autofunction:: pymanoid.gui.draw_point
.. autofunction:: pymanoid.gui.draw_points
.. autofunction:: pymanoid.gui.draw_trajectory
.. autofunction:: pymanoid.gui.draw_wrench

Convex polyhedra
================

`Polyhedra <https://en.wikipedia.org/wiki/Convex_polytope>`_ correspond to the
sets of linear inequality constraints applied to the system, for instance the
support area in which the zero-tilting moment point (ZMP) of a legged robot
should lie to avoid breaking contacts with the ground.

.. autofunction:: pymanoid.gui.draw_2d_cone
.. autofunction:: pymanoid.gui.draw_cone
.. autofunction:: pymanoid.gui.draw_horizontal_polygon
.. autofunction:: pymanoid.gui.draw_polygon
.. autofunction:: pymanoid.gui.draw_polytope

Drawers
=======

.. autoclass:: pymanoid.gui.WrenchDrawer
    :members:

.. autoclass:: pymanoid.gui.PointMassWrenchDrawer
    :members:

.. autoclass:: pymanoid.gui.RobotWrenchDrawer
    :members:

.. autoclass:: pymanoid.gui.StaticEquilibriumWrenchDrawer
    :members:

.. autoclass:: pymanoid.gui.TrajectoryDrawer
    :members:
