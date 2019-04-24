**************************
Walking pattern generation
**************************

Walking pattern generation converts a high-level objective such as "go through
this sequence of footsteps" to a time-continuous joint trajectory. For
position-controlled robots such as those simulated in pymanoid, joint
trajectories are computed by inverse kinematics from intermediate task targets.
The two main targets for walking are the swing foot and center of mass.

Swing foot trajectory
=====================

The foot in the air during a single-support phase is called the *swing foot*.
Its trajectory can be implemented by `spherical linear interpolation
<https://en.wikipedia.org/wiki/Slerp>`_ for the orientation and `polynomial
interpolation <https://en.wikipedia.org/wiki/Polynomial_interpolation>`_ for
the position of the foot in the air.

.. autoclass:: pymanoid.swing_foot.SwingFoot
    :members:

Linear model predictive control
===============================

Linear model predictive control [Wieber06]_ generates a dynamically-consistent
trajectory for the center of mass (COM) while walking. It applies to walking
over a flat floor, where the assumptions of the linear inverted pendulum mode
(LIPM) [Kajita01]_ can be applied.

.. autoclass:: pymanoid.mpc.LinearPredictiveControl
    :members:

Nonlinear predictive control
=============================

The assumptions of the LIPM are usually too restrictive to walk over uneven
terrains. In this case, one can turn to more general (nonlinear) model
predictive control. A building block for nonlinear predictive control of the
center of mass is implemented in the
:class:`pymanoid.centroidal.COMStepTransit` class, where a COM trajectory is
constructed from one footstep to the next by solving a `nonlinear program
<numerical-optimization.html#nonlinear-programming>`_ (NLP).

.. autoclass:: pymanoid.centroidal.COMStepTransit
    :members:
