# openravepypy

Python bindings for OpenRAVE. 

A wrapper to `openravepy` with some more pythonic constructions. The library
includes some features useful for humanoid motion planning that are not present
in OpenRAVE, such as:

- Computing the Angular Momentum, its jacobian or (pseudo-)hessian
- Computing the Center-of-Mass position, velocity, jacobian or hessian
- Computing the ZMP (Zero-tipping Moment Point)
- Numerical IK solver using [CVXOPT](http://cvxopt.org/index.html)
  (allows for task-based whole-body optimization, does not require IKFast)

This repository is part of my working code; read: unstable `;)`

## Installation

From the top folder, run: `sudo python setup.py install`
