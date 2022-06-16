# pymanoid

[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](https://opensource.org/licenses/GPL-3.0)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://scaron.info/doc/pymanoid/)
![Status](https://img.shields.io/badge/status-archive-lightgrey.svg)

Humanoid robotics controller prototyping environment based on [OpenRAVE](https://github.com/rdiankov/openrave).

## Features

### Whole-body inverse kinematics

<a href="https://github.com/tasts-robots/pink">
    <img src="https://user-images.githubusercontent.com/1189580/172797197-9aa46561-cfaa-4046-bd60-f681d85b055d.png" align="right" height=100>
</a>

- Check out **🟣 [Pink](https://github.com/tasts-robots/pink)** for a next-generation implementation of this IK as a library 📚
- Based on a [weight-prioritized multi-task formulation](https://scaron.info/robot-locomotion/inverse-kinematics.html) of differential IK
- Jacobians and Hessians for center of mass (CoM) and angular momentum tasks

### Contact stability

- [Wrench friction cones](http://www.roboticsproceedings.org/rss11/p28.pdf) for general multi-contact motions
- [Multi-contact ZMP support areas](https://hal.archives-ouvertes.fr/hal-02108589/document) for locomotion
- [CoM acceleration cones](https://hal.archives-ouvertes.fr/hal-01349880/document) for locomotion (conservative)
- [Robust CoM static-equilibrium polytope](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01477362/document) for posture generation (conservative)

### Model predictive control

- [Linear model predictive control](https://scaron.info/publications/humanoids-2016.html) (LMPC) for locomotion
- [Nonlinear model predictive control](https://scaron.info/publications/iros-2017.html) (NMPC) for locomotion
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
- Tutorial: [Prototyping a walking pattern generator](https://scaron.info/robot-locomotion/prototyping-a-walking-pattern-generator.html)

## Installation

The following instructions were verified on Ubuntu 14.04:

- Install OpenRAVE: here are [instructions for Ubuntu 14.04](https://scaron.info/robot-locomotion/installing-openrave-on-ubuntu-14.04.html) as well as [for Ubuntu 16.04](https://scaron.info/robot-locomotion/installing-openrave-on-ubuntu-16.04.html)
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

## Citing pymanoid

I developed pymanoid during my PhD studies and share it in the hope it can be useful to others. If it helped you in your research, feel free to show it with academic kudos, *a.k.a.* citations :-)

```bibtex
@phdthesis{caron2016thesis,
    title = {Computational Foundation for Planner-in-the-Loop Multi-Contact Whole-Body Control of Humanoid Robots},
    author = {Caron, St{\'e}phane},
    year = {2016},
    month = jan,
    school = {The University of Tokyo},
    url = {https://scaron.info/papers/thesis.pdf},
    doi = {10.15083/00074003},
}
```
