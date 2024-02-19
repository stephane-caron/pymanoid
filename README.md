# pymanoid

[![License](https://img.shields.io/badge/License-GPLv3-green.svg)](https://opensource.org/licenses/GPL-3.0)
[![Documentation](https://img.shields.io/badge/docs-online-brightgreen?logo=read-the-docs&style=flat)](https://scaron.info/doc/pymanoid/)
![Status](https://img.shields.io/badge/status-archive-lightgrey.svg)

Humanoid robotics controller prototyping environment based on [OpenRAVE](https://github.com/rdiankov/openrave).

> ⚠️ This project is **archived**. Feel free to look at the code, but don't expect support to install and run it.

Most of the project's functionality has been ported to follow-up libraries that are maintained and easier to install.

## Follow-up software

<a href="https://github.com/stephane-caron/pink">
    <img src="https://user-images.githubusercontent.com/1189580/172797197-9aa46561-cfaa-4046-bd60-f681d85b055d.png" align="right" height=100>
</a>

- [pink](https://github.com/stephane-caron/pink): inverse kinematics in Python based on [Pinocchio](https://github.com/stack-of-tasks/pinocchio/)
- [pypoman](https://github.com/stephane-caron/pypoman): polyhedral projection functions used to compute contact inequality constraints
- [qpmpc](https://github.com/stephane-caron/qpmpc): linear model predictive control in Python
- [qpsolvers](https://github.com/qpsolvers/qpsolvers): interfaces to quadratic programming solvers in Python
- [vhip\_light](https://github.com/stephane-caron/vhip_light): variable-height inverted pendulum balancing in Python

## Features

### Contact stability

- [Wrench friction cones](http://www.roboticsproceedings.org/rss11/p28.pdf) for general multi-contact motions
- [Multi-contact ZMP support areas](https://hal.archives-ouvertes.fr/hal-02108589/document) for locomotion
- [CoM acceleration cones](https://hal.archives-ouvertes.fr/hal-01349880/document) for locomotion (conservative)
- [Robust CoM static-equilibrium polytope](https://hal-lirmm.ccsd.cnrs.fr/lirmm-01477362/document) for posture generation (conservative)

### Model predictive control

- [Linear model predictive control](https://hal.archives-ouvertes.fr/hal-01349880/document) (LMPC) for locomotion
- [Nonlinear model predictive control](https://hal.archives-ouvertes.fr/hal-01481052/document) (NMPC) for locomotion

### Inverse kinematics

- Whole-body IK based on the [weight-prioritized multi-task formulation](https://scaron.info/robot-locomotion/inverse-kinematics.html)
- Jacobians and Hessians for center of mass (CoM) and angular momentum tasks

### Geometry and optimization toolbox

- Interfaces to polyhedral geometry: double description, polytope projection
- Interfaces for numerical optimization solvers: LP, QP and NLP

## Use cases

<img src="doc/src/images/logo.png" width="350" align="right" />

- [Walking pattern generation over uneven terrains](https://github.com/stephane-caron/capture-walkgen) based on capturability of the variable-height inverted pendulum model
- [Nonlinear model predictive control](https://github.com/stephane-caron/fip-walkgen) using a direct transcription of centroidal dynamics
- [Linearized model predictive control](https://github.com/stephane-caron/multi-contact-walkgen) using a conservative linearization of CoM acceleration cones
- [Multi-contact ZMP support areas](https://github.com/stephane-caron/multi-contact-zmp) for locomotion in multi-contact scenarios (including hand contacts)
- [Humanoid stair climbing](https://github.com/stephane-caron/quasistatic-stair-climbing) demonstrated on the HRP-4 robot

## Getting started

- [Documentation](https://scaron.info/doc/pymanoid/)
- [FAQ](https://github.com/stephane-caron/pymanoid/wiki/Frequently-Asked-Questions)
- [Examples](https://github.com/stephane-caron/pymanoid/tree/master/examples)
- Tutorial: [Prototyping a walking pattern generator](https://scaron.info/robot-locomotion/prototyping-a-walking-pattern-generator.html)

## Citing pymanoid

I developed pymanoid during my PhD studies and share it in the hope it can be useful to others. If it helped you in your research, please cite it *e.g.* as follows:

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

## Installation

It is not recommended to try to install this library as it is not maintained and relies on deprecated software. Yet, If you are digging into the archives, you can find instructions for Ubuntu 14.04 in [the wiki](https://github.com/stephane-caron/pymanoid/wiki/Installation-instructions).
