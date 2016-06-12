# Examples

## inverse\_kinematics.py

<img align="right" src="https://raw.githubusercontent.com/stephane-caron/pymanoid/master/examples/images/inverse_kinematics.png" width="250" />

This is a toy example showing how to use the IK for whole-body motions. It
loads the JVRC-1 model and generates a posture where the robot has its two feet
on pre-defined contacts. Then, its COM tracks the motion of a virtual box using
the ``step_ik()`` function.

To get this example running, you will need a copy or symbolic link of the
[openrave\_models](https://github.com/stephane-caron/openrave_models)
repository in the local folder. Alternatively, you can clone the repository
anywhere and set the correct path in the ``env_file`` variable of the script.

## static\_equilibrium\_polygon.py

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
