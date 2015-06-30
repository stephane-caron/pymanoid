# openravepypy

Python bindings for OpenRAVE. Actually, a wrapper to `openravepy` with some
more pythonic constructions.

This repository is part of my working code, therefore completely unstable.

## Features

The library includes some features useful for humanoid motion planning that are
not present in OpenRAVE, such as:

- calculation of the ZMP (Zero-tipping Moment Point), COM (Center Of Mass) and
  angular momentum, as well as their jacobians and hessians
- a small numerical IK solver using [CVXOPT](http://cvxopt.org/index.html),
  allowing for task-based whole-body resolution (it doesn't require IKFast)
