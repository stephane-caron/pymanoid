# pymanoid

<img src="https://scaron.info/images/ijhr-2016.png" width="350" align="right" />

Python library for humanoid robotics in OpenRAVE.

## Features

- Inverse kinematics (IK) implementing a [weight-prioritized multi-task
  solver](https://hal.archives-ouvertes.fr/hal-01247118). Also known as
  *generalized IK* or *whole-body IK*, this approach allows general objectives
  such as center-of-mass (COM) or angular-momentum tracking.
- Jacobians and hessians for the COM, zero-tilting moment point (ZMP), angular
  momentum, ...
- Contact-stability criteria: contact wrench cones, stability polygons, ...

## Installation

- First, [install OpenRAVE](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html).
- Install all Python dependencies:
```
sudo apt-get install cython python python-dev python-pip python-scipy
sudo pip install quadprog pycddlib
```
- Finally, if you wish to install the library system-wide:
```
sudo python setup.py install
```

The preferred QP solver is [quadprog](https://github.com/rmcgibbo/quadprog).
Alternatively, *pymanoid* will try to use [CVXOPT](http://cvxopt.org) or
[qpOASES](https://projects.coin-or.org/qpOASES) if they are installed.

## Usage

See the [examples](/examples) folder for some sample test cases.
