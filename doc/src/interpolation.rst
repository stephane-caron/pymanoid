*************
Interpolation
*************

Interpolation is the operation of computing trajectories that connect an
initial state (position and its derivatives such as velocity and acceleration)
to a desired one.

Position
========

In the absence of inequality constraints, interpolation for positions can be
solved by `polynomial interpolation
<https://en.wikipedia.org/wiki/Polynomial_interpolation>`_. The following
functions and classes provide simple interfaces for this common operation.

.. autofunction:: pymanoid.interp.interpolate_cubic_bezier

.. autofunction:: pymanoid.interp.interpolate_cubic_hermite

.. autoclass:: pymanoid.interp.LinearPosInterpolator
    :members:

.. autoclass:: pymanoid.interp.CubicPosInterpolator
    :members:

.. autoclass:: pymanoid.interp.QuinticPosInterpolator
    :members:

Orientation
===========

In the absence of inequality constraints, interpolation for orientations can be
solved by `spherical linear interpolation
<https://en.wikipedia.org/wiki/Slerp>`_. The following functions and classes
provide simple interfaces for this common operation. They compute *poses*:
following OpenRAVE terminology, the pose of a rigid body denotes the 7D vector
of its 4D orientation quaternion followed by its 3D position coordinates.

.. autofunction:: pymanoid.interp.interpolate_pose_linear

.. autofunction:: pymanoid.interp.interpolate_pose_quadratic

.. autoclass:: pymanoid.interp.LinearPoseInterpolator
    :members:

.. autoclass:: pymanoid.interp.CubicPoseInterpolator
    :members:

.. autoclass:: pymanoid.interp.PoseInterpolator
    :members:

.. autoclass:: pymanoid.interp.QuinticPoseInterpolator
    :members:
