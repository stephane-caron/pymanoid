************************
Model Predictive Control
************************

Linear Predictive Control
=========================

.. automodule:: pymanoid.mpc
    :members:
    :undoc-members:

Centroidal Predictive Control
=============================

A building block for centroidal predictive control is implemented in the
:class:`pymanoid.centroidal.COMStepTransit` class, where a COM trajectory is
constructed from one footstep to the next by solving a `nonlinear program
<numerical-optimization.html#nonlinear-programming>`_ (NLP).

.. autoclass:: pymanoid.centroidal.COMStepTransit
    :members:
    :undoc-members:
