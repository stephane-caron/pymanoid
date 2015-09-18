# openravepypy

QP-based inverse kinematics, contact double-description and more for OpenRAVE.

A wrapper to `openravepy` with some more pythonic constructions. The library
includes some features useful for humanoid motion planning that are not present
in OpenRAVE, such as:

- QP-based Inverse Kinematics using [CVXOPT](http://cvxopt.org/index.html)
  (allows for task-based whole-body optimization, does not require IKFast)
- Contact double-description routines for whole-body multi-contact planning
- Jacobians and hessians for the center of mass, zero-tilting moment point,
  angular momentum, ...

**Disclaimer:** this repository is part of my working code (means: "unstable").

## Installation

From the top folder, run: `sudo python setup.py install`
