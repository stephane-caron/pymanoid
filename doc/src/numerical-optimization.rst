**********************
Numerical optimization
**********************

Quadratic programming
=====================

Quadratic programming is a class of numerical optimization problems that can be
solved efficiently. It is used for instance by the `IK solver
<inverse-kinematics.html#solver>`_ to solve whole-body control by inverse
kinematics.

.. autofunction:: pymanoid.qpsolvers.solve_qp

Nonlinear programming
=====================

Nonlinear programming is a catch-all expression for numerical optimization
problems that don't have any particular structure, such as *convexity*,
allowing them to be solved more efficiently by dedicated methods. It is used
for instance in `nonlinear model predictive control
<walking-pattern-generation.html#nonlinear-predictive-control>`_ to compute
locomotion trajectories over uneven terrains.

.. autoclass:: pymanoid.nlp.NonlinearProgram
    :members:
