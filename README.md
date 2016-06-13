# pymanoid

Python library for humanoid robotics in OpenRAVE.

## Features

<img src="https://scaron.info/images/ijhr-2016.png" width="350" align="right" />

- Custom inverse kinematics (IK) solver, slower than IKFast but taking into
  account redundancy, high-DOF systems and general objectives such as
  center-of-mass position or angular-momentum tracking
- Jacobians and hessians for the center of mass, ZMP and angular momentum
- Calculation of contact-stability criteria: contact wrench cones, stability
  polygons, etc. 

## Dependencies

- [CVXOPT](http://cvxopt.org/) used for its QP and LP solvers
- [OpenRAVE](https://github.com/rdiankov/openrave) used for forward kinematics and visualization ([installation instructions](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html))
- [pycddlib](https://pycddlib.readthedocs.org/en/latest/) used for cone duality calculations

## Installation

On Ubuntu 14.04, run the following from the top-level directory:

```
sudo apt-get install cython python python-dev python-pip python-scipy
sudo pip install cvxopt pycddlib
sudo python setup.py install
```
