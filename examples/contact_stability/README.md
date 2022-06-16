# Contact-stability conditions

Contact-stability areas and volumes are conditions used to prevent contacts
from slipping or detaching during motion.

## Wrench friction cone

The ``wrench_friction_cone.py`` script computes the wrench friction cone of
the robot's contacts, as described in [this
paper](http://www.roboticsproceedings.org/rss11/p28.pdf). Wrench friction cones
are 6D polyhedral convex cones that characterize feasible contact wrenches,
that is to say, wrenches that can be realized during contact-stable motions.
They are a 6D generalization of Coulomb friction cones, and can also be used to
encode other power limitations such as maximum joint torques.

## CoM static-equilibrium polygon

The ``com_static_polygon.py`` script illustrates the polygon of CoM positions
that the robot can hold in static equilibirum, as derived in [this
paper](https://doi.org/10.1109/TRO.2008.2001360). You can move contacts by
selecting them in the OpenRAVE GUI. Contact wrenches are computed at each
contact to support the robot in static-equilibrium. Try moving the blue box (in
the plane above the robot) around, and see what happens when it exits the
polygon.

## Multi-contact ZMP support areas

Th ``zmp_support_area.py`` script displays the ZMP support area under a given
set of contacts. The derivation of this area is detailed in [this
paper](https://hal.archives-ouvertes.fr/hal-02108589/document). It depends on both contact
locations and the position of the center of mass, so when you move it or its
projection (blue box) you will see the blue area change as well.

## CoM acceleration cone

The ``com_accel_cone.py`` script displays the cone of CoM accelerations that
the robot can execute while keeping contacts, as derived in [this
paper](https://hal.archives-ouvertes.fr/hal-01349880/document). Like ZMP support
areas, this cone depends on both contact locations and the position of the
center of mass, so that when you move it or its projection (blue box) you will
see its shape change as well.

## CoM robust static-equilibrium polytope

The ``com_robust_static_polytope.py`` example generalizes the previous one when
there are additional constraints on the robot, such as external forces applied
on the robot as described in [this
paper](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01477362/document). In this case,
the upright prism of the static-equilibrium polygon generalizes into a
*polytope* of sustainable CoM positions (an intersection of slanted
static-equilibrium prisms). This example illustrates how to compute this
polytope using the [StabiliPy](https://github.com/haudren/stabilipy) library.
Try moving contacts around to see what happens.
