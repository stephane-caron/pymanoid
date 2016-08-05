# pymanoid

Python library for humanoid robotics. Extends OpenRAVE with:

- Primitives for contacts, stances, point-mass systems, etc.
- Jacobians and hessians for contacts, center of mass and angular momentum
- **Whole-body Inverse Kinematics** based on the [weight-prioritized
  multi-task framework](https://hal.archives-ouvertes.fr/hal-01247118)
- **Contact stability criteria:** Contact Wrench Cone, stability polygons,
  computation of supporting contact forces, etc.
- Backends for Linear Programming and Quadratic Programming (QP) solvers
- Drawing primitives for 2D and 3D polyhedra (polygons, polytopes, cones, ...)

<img src="https://scaron.info/images/ijhr-2016.png" width="350" align="right" />

## Use cases

- [Multi-contact Walking Pattern Generation](https://scaron.info/research/pre-print-2016-1.html)
  based on Model-Preview Control of the 3D acceleration of the center of mass
- [Generalized ZMP support areas](https://scaron.info/research/arxiv-2015.html)
  for locomotion in multi-contact scenarios, including e.g. hand contacts
- [Humanoid Stair Climbing](https://scaron.info/research/ijhr-2016.html)
  demonstrated on the real HRP-4 robot

## Installation

First, install OpenRAVE (here are some [instructions for Ubuntu
14.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html)).
Next, install all Python dependencies with:
```
sudo apt-get install cython python python-dev python-pip python-scipy
sudo pip install quadprog pycddlib
```
Finally, if you wish to install the library system-wide:
```
sudo python setup.py install
```
The preferred QP solver is [quadprog](https://github.com/rmcgibbo/quadprog).
Alternatively, the library will try to use [CVXOPT](http://cvxopt.org) if it is
installed.

## Usage

See the [examples](/examples) folder for some sample test cases. For a more
elaborate use case, see [3d-mpc](https://github.com/stephane-caron/3d-mpc).
