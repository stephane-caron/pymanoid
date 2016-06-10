# Examples

## inverse\_kinematics.py

This is a small example on how to use the IK for whole-body motions. To get it
running, you will need a copy or symbolic link of the
[openrave\_models](https://github.com/stephane-caron/openrave_models)
repository in the local folder. Alternatively, you can clone the repository
anywhere and set the correct path in the ``env_file`` variable of the script.

## static\_equilibrium\_polygon.py

In this example, we display the static-equilibrium COM polygon (in magenta) for
a given set of contacts.
    
You can move contacts by selecting them in the OpenRAVE GUI. The robot IK is
servoed to their positions. Type ``recompute_polygon()`` to recompute the COM
polygon after moving contacts.

<div style="float: right; margin-left: 1em">
<img src="https://raw.githubusercontent.com/stephane-caron/pymanoid/master/examples/images/static_equilibrium_polygon.png" height="300" />
</div>

To illustrate the validity of this polygon, contact forces are computed that
support the equilibrium position represented by the blue box (which acts like a
COM position). Try moving this box around, and see what happens when it exits
the polygon.
