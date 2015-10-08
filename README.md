# pymanoid

Python library for humanoid robotics in OpenRAVE.

**Disclaimer:** this repository is part of my working code (means: "unstable").

## Features

- QP-based Inverse Kinematics using [CVXOPT](http://cvxopt.org/index.html)
  (allows for task-based whole-body optimization, does not require IKFast)
- Contact double-description method for whole-body multi-contact planning
- Jacobians and hessians for the center of mass, ZMP, angular momentum, ...

## Installation

From the top folder, run: `sudo python setup.py install`
