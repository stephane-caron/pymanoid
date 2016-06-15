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
- Make sure you have all Python dependencies:

```
sudo apt-get install cython python python-dev python-pip python-scipy
```

- Install [CVXOPT](http://cvxopt.org/) and [pycddlib](https://pycddlib.readthedocs.org/en/latest/):

```
sudo pip install cvxopt pycddlib
```

- To use in a single directory, make a symbolic link to the ``pymanoid``
  sub-folder of the cloned repository
- To install the library system-wide:

```
sudo python setup.py install
```
