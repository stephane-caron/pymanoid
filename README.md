# pymanoid

Humanoid robotics controller prototyping environment based on
[OpenRAVE](https://github.com/rdiankov/openrave). Includes:

- Whole-body inverse kinematics (IK) based on the 
  [weight-prioritized multi-task framework](https://scaron.info/teaching/inverse-kinematics.html)
- Contact-stability areas and volumes: 
  [multi-contact ZMP support areas](https://scaron.info/publications/tro-2016.html),
  [CoM acceleration cones](https://scaron.info/publications/humanoids-2016.html), etc.
- [Linear](https://scaron.info/publications/humanoids-2016.html) and
  [Nonlinear Model Predictive Control](https://scaron.info/publications/iros-2017.html)
  (MPC) for locomotion
- Jacobians and Hessians for center of mass (CoM) and angular momentum
- Interfaces to polyhedral geometry and numerical optimization (LP, QP and NLP) solvers

## Use cases

<img src="doc/src/images/logo.png" width="350" align="right" />

- [Walking pattern generation over uneven terrains](https://github.com/stephane-caron/capture-walkgen)
  based on capturability of the variable-height inverted pendulum model
- [Nonlinear model predictive control](https://github.com/stephane-caron/fip-walkgen)
  using a direct transcription of centroidal dynamics
- [Linearized model predictive control](https://github.com/stephane-caron/multi-contact-walkgen)
  using a conservative linearization of CoM acceleration cones
- [Multi-contact ZMP support areas](https://github.com/stephane-caron/multi-contact-zmp)
  for locomotion in multi-contact scenarios (including hand contacts)
- [Humanoid stair climbing](https://github.com/stephane-caron/quasistatic-stair-climbing)
  demonstrated on the HRP-4 robot

## Getting started

- [Installation instructions](#installation)
- [Documentation](https://scaron.info/doc/pymanoid/) ([PDF](https://scaron.info/doc/pymanoid/pymanoid.pdf))
- [FAQ](https://github.com/stephane-caron/pymanoid/wiki/Frequently-Asked-Questions)
- [Examples](/examples)
- Tutorial: [Prototyping a walking pattern generator](https://scaron.info/teaching/prototyping-a-walking-pattern-generator.html)

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
