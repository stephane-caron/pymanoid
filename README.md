# openravepypy

Python bindings for OpenRAVE. Actually, a wrapper to `openravepy` with some
more pythonic constructions.

This repository is part of my working code, thus completely unstable `:)`

## Features

The library includes some features useful for humanoid motion planning that are
not present in OpenRAVE, such as:

- Calculation of the ZMP (Zero-tipping Moment Point), center of mass and
  angular momentum, as well as their jacobians and hessians
- Numerical IK solver using [CVXOPT](http://cvxopt.org/index.html)
  allowing for task-based whole-body resolution (it doesn't require IKFast)
