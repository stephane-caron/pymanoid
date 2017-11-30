# Contact-stability conditions

Contact-stability areas and volumes are conditions used to prevent contacts
from slipping or detaching during motion. These examples illustrate the
static-equilibrium COM polygon, multi-contact ZMP support areas and the 3D cone
of COM accelerations.

## COM static-equilibrium polygon

<img align="right" src="../../doc/source/images/static_equilibrium_polygon.png" width="300" />
    
You can move contacts by selecting them in the OpenRAVE GUI. Contact wrenches
are computed at each contact to support the robot in static-equilibrium. Try
moving the blue box (in the plane above the robot) around, and see what happens
when it exits the polygon.

## Multi-contact ZMP support areas

The definition and calculation of the ZMP support area is detailed in [this
paper](https://scaron.info/research/tro-2016.html). These areas depend on
contact locations and on the position of the center of mass, so when you move
it or its projection (blue box) you will see the blue area change as well.

## COM acceleration cones

The definition and calculation of pendular COM acceleration cones is detailed
in [this paper](https://scaron.info/research/humanoids-2016.html). These cones
depend on contact locations and on the position of the center of mass, so when
you move it or its projection (blue box) you will see the red cone change as
well.
