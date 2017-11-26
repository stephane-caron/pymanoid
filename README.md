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

<img src="doc/source/images/logo.png" width="350" align="right" />

- [Dynamic walking over rough terrains](https://github.com/stephane-caron/dynamic-walking)
  based on nonlinear model predictive control of the floating-base inverted
  pendulum model
- [Multi-contact walking pattern generation](https://github.com/stephane-caron/3d-walking-lmpc)
  based on linear model predictive control of 3D CoM accelerations
- [Multi-contact ZMP support areas](https://github.com/stephane-caron/multi-contact-zmp)
  for locomotion in multi-contact scenarios (including hand contacts)
- [Humanoid stair climbing](https://github.com/stephane-caron/stair-climbing)
  demonstrated on a Kawada HRP-4 robot

## Getting started

- [Installation instructions](#installation)
- [Documentation](https://scaron.info/doc/pymanoid/)
- [FAQ](#frequently-asked-questions)
- [Examples](/examples)

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
Finally, clone the repository, and run the setup script:
```
git clone https://github.com/stephane-caron/pymanoid.git && cd pymanoid
python setup.py build
python setup.py install --user
```

### Optional

For nonlinear optimal control, you will need to [install
CasADi](https://github.com/casadi/casadi/wiki/InstallationLinux), preferably
from source to install the MA27 linear solver as well.

Some minor functions to manipulate polygons may also require two small
libraries: ``sudo pip install pyclipper``.

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
motion. This check is performed by the ``find_supporting_wrenches()``
function of a [ContactSet](/pymanoid/contact.py).

**Q: How can I record a video of my simulation?**

For a quick and dirty solution, you can record your whole desktop using e.g.
[kazam](https://github.com/sconts/kazam). However, with this approach the video
time will be your system time, potentially slowed down by all other processes
running on your machine (including `kazam` itself).

To record a video synchronized with your simulation time, call
`sim.record("filename.mp4")`. This will schedule an extra `CameraRecorder`
process that takes a capture of your simulation window after each step. You can
then run your simulation as usual. Once your simulation is over, call the
`make_pymanoid_video.sh` script created in the current folder. It will
reassemble screenshots into a properly timed video `filename.mp4`.

**Q: the video script returns an error "width/height not divisible by 2".**

Bad choice of window size ;) You will need to crop the PNG files in the
`pymanoid_rec/` temporary folder. For instance, if the resolution of these
files is 1918x1059, `cd` to that folder and run `mogrify -crop 1918x1058+0+0
*.png`.
