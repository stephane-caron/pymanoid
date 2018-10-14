# pymanoid

Python library for humanoid robotics based on
[OpenRAVE](https://github.com/rdiankov/openrave):

- Whole-body inverse kinematics (IK) based on the [weight-prioritized
  multi-task framework](https://scaron.info/teaching/inverse-kinematics.html)
- Contact-stability areas and volumes: [multi-contact ZMP
  support areas](https://scaron.info/research/tro-2016.html), [CoM acceleration
  cones](https://scaron.info/research/humanoids-2016.html), etc.
- Linear Model Predictive Control (MPC) and state machines for locomotion
- Jacobians and Hessians for center of mass (CoM) and angular momentum
- Types and algorithms to manipulate polytopes and polyhedral cones
- Interface to linear programming (LP) and quadratic programming (QP) solvers

## Use cases

<img src="doc/src/images/logo.png" width="350" align="right" />

- [3D bipedal walking over uneven terrains](https://github.com/stephane-caron/capture-walking) 
  based on a capturability analysis of the 3D inverted pendulum model
- [Nonlinear model predictive control](https://github.com/stephane-caron/fip-walking)
  using a direct transcription of centroidal dynamics
- [Linearized model predictive control](https://github.com/stephane-caron/3d-walking-lmpc)
  using a conservative linearization of CoM acceleration cones
- [Multi-contact ZMP support areas](https://github.com/stephane-caron/multi-contact-zmp)
  for locomotion in multi-contact scenarios (including hand contacts)
- [Humanoid stair climbing](https://github.com/stephane-caron/stair-climbing)
  demonstrated on the HRP-4 humanoid robot

## Getting started

- [Installation instructions](#installation)
- Documentation: [html](https://scaron.info/doc/pymanoid/) or [pdf](https://scaron.info/doc/pymanoid/pymanoid.pdf)
- [FAQ](https://github.com/stephane-caron/pymanoid/wiki/Frequently-Asked-Questions)
- [Examples](/examples)

## Installation

The following instructions were verified on Ubuntu 14.04:

- Install OpenRAVE: here are [instructions for Ubuntu 14.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html) as well as [for Ubuntu 16.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html)
- Install Python dependencies: 
```
sudo apt-get install cython libglpk-dev python python-dev python-pip python-scipy python-simplejson
```
- Install the LP solver: ``CVXOPT_BUILD_GLPK=1 pip install cvxopt --user``
- Install the QP solver: ``pip install quadprog --user``
- For polyhedral computations (optional): ``pip install pycddlib --user``

Finally, clone this repository and run the setup script:
```
git clone --recursive https://github.com/stephane-caron/pymanoid.git
cd pymanoid
python setup.py build
python setup.py install --user
```

### Optional

For nonlinear numerical optimization, you will need to [install
CasADi](https://github.com/casadi/casadi/wiki/InstallationLinux), preferably
from source with the MA27 linear solver.
