# pymanoid

<img src="https://scaron.info/images/ijhr-2016.png" width="400" align="right" />

Python library for humanoid robotics in OpenRAVE.

Features:
- Numerical Inverse Kinematics solver (slower than IKFast, but deals with
  redundancy, high-DOF systems and general objectives such as center-of-mass or
  angular momentum)
- Jacobians and hessians for the center-of-mass, ZMP and angular momentum
- Double-description method for multi-contact stability

This repository is part of my working code. Classes and function prototypes may
change at any time without notice.

## Dependencies

- [CVXOPT](http://cvxopt.org/) used for its QP and LP solvers
- [OpenRAVE](https://github.com/rdiankov/openrave) used for forward kinematics and visualization. See e.g. the following [installation instructions](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html).
- [pycddlib](https://pycddlib.readthedocs.org/en/latest/) used for cone duality
  calculations.

## Installation

First, install dependencies from your package manager. On Ubuntu 14.04:

```
sudo apt-get install cython python python-dev python-pip python-scipy
sudo pip install cvxopt pycddlib
```

Then, from the top folder, run: `sudo python setup.py install`.
