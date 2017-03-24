# pymanoid

Python library for humanoid robotics based on
[OpenRAVE](https://github.com/rdiankov/openrave):

- Whole-body inverse kinematics (IK) based on the [weight-prioritized
  multi-task framework](https://scaron.info/teaching/inverse-kinematics.html)
- Contact-stability areas and volumes: [multi-contact ZMP
  support areas](https://scaron.info/research/tro-2016.html), [COM acceleration
  cones](https://scaron.info/research/humanoids-2016.html), etc.
- Linear Model Predictive Control (MPC) and state machines for locomotion
- Jacobians and Hessians for center of mass (COM) and angular momentum
- Types and algorithms to manipulate polytopes and polyhedral cones
- Interface to linear programming (LP) and quadratic programming (QP) solvers

<img src="https://scaron.info/images/ijhr-2016.png" width="350" align="right" />

## Use cases

- [Multi-contact Walking Pattern Generation](https://scaron.info/research/humanoids-2016.html)
  based on Model-Preview Control of the 3D acceleration of the center of mass
- [Generalized ZMP support areas](https://scaron.info/research/tro-2016.html)
  for locomotion in multi-contact scenarios, including e.g. hand contacts
- [Humanoid Stair Climbing](https://scaron.info/research/ijhr-2016.html)
  demonstrated on the real HRP-4 robot

## Getting started

- [Installation instructions](#installation)
- [FAQ](#frequently-asked-questions)
- [Examples](/examples)
- [API documentation](https://scaron.info/doc/pymanoid/)

## Installation

First, you will need to install
[OpenRAVE](https://github.com/rdiankov/openrave). Here are some [instructions
for Ubuntu
14.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-14.04.html)
and [Ubuntu
16.04](https://scaron.info/teaching/installing-openrave-on-ubuntu-16.04.html).

Next, install all Python dependencies with:
```
sudo apt-get install cython libglpk-dev python python-dev python-pip python-scipy python-simplejson
sudo pip install quadprog pycddlib
sudo CVXOPT_BUILD_GLPK=1 pip install cvxopt
```
Finally, clone the repository, and run the setup script if you wish to install
the library system-wide:
```
git clone https://github.com/stephane-caron/pymanoid.git && cd pymanoid
sudo python setup.py install
```

## Frequently Asked Questions

**Q: Do you implement dynamics simulation in Pymanoid? If yes, could you give
me some pointers? If no, how do you verify the stability of the robot?**

Forward dynamics need a reaction-force model, which is a tricky thing to do in
rigid body dynamics. This stems from an essential "contradiction" of the model:
physically, reaction forces depend on local deformations between bodies in
contact, while the main assumption of rigid body dynamics is that bodies are
not deformable. To overcome this, two main approaches have been explored:
regularized reaction-force models (a.k.a. "jedi" physics) and non-smooth
approaches. Both have pros and cons in terms of realism and numerical
integration. For more details, check out the Wikipedia page on [contact
dynamics](https://en.wikipedia.org/wiki/Contact_dynamics).

Pymanoid does not provide forward dynamics. The stability that is checked in
simulations is a feasibility criterion called [contact
stability](https://scaron.info/teaching/contact-stability.html), namely, that
at each timestep there exists feasible contact forces that support the robot
motion. To perform this check, you can instantiate a [wrench
drawer](/pymanoid/drawers.py) or call the ``find_supporting_wrenches()``
function from a [ContactSet](/pymanoid/contact_set.py).
