************************
Model predictive control
************************

Linear predictive control
=========================

.. automodule:: pymanoid.mpc
    :members:
    :undoc-members:

Nonlinear predictive control
=============================

A building block for nonlinear predictive control of the center of mass is
implemented in the :class:`pymanoid.centroidal.COMStepTransit` class, where a
COM trajectory is constructed from one footstep to the next by solving a
`nonlinear program <numerical-optimization.html#nonlinear-programming>`_ (NLP).

.. autoclass:: pymanoid.centroidal.COMStepTransit
    :members:
    :undoc-members:
