# pymanoid

<img src="https://scaron.info/images/ijhr-2016.png" width="350" align="right" />

Python library for humanoid robotics in OpenRAVE.

## Features

- Custom inverse kinematics (IK) solver, slower than IKFast but taking into
  account redundancy, high-DOF systems and general objectives such as
  center-of-mass position or angular-momentum tracking
- Jacobians and hessians for the center of mass, ZMP and angular momentum
- Calculation of contact-stability criteria: contact wrench cones, stability
  polygons, etc. 

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
