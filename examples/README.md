# Examples

## Inverse Kinematics

<img align="right" src="https://raw.githubusercontent.com/stephane-caron/pymanoid/master/examples/images/inverse_kinematics.png" width="250" />

This example shows how to use the IK to generate whole-body motions. It loads
(and downloads, if necessary) the
[JVRC-1](https://github.com/stephane-caron/openrave_models/tree/master/JVRC-1)
model, then generates a posture where the robot has its two feet on pre-defined
contacts. Next, it tracks a reference COM motion given by the green virtual box
using the ``step_ik()`` function.

Once the demonstration is over, the IK thread is launched. You can then move
the virtual box directly in the OpenRAVE GUI.

## Static-equilibrium Polygon

<img align="right" src="https://raw.githubusercontent.com/stephane-caron/pymanoid/master/examples/images/static_equilibrium_polygon.png" width="250" />

In this example, we display the static-equilibrium COM polygon (in magenta) for
a given set of contacts.
    
You can move contacts by selecting them in the OpenRAVE GUI. The robot IK is
servoed to their positions. Type ``recompute_polygon()`` to recompute the COM
polygon after moving contacts.

To illustrate the validity of this polygon, contact forces are computed that
support the equilibrium position represented by the blue box (which acts like a
COM position). Try moving this box around, and see what happens when it exits
the polygon.
