# pymanoid

Python library for humanoid robotics in OpenRAVE.

**Disclaimer:** this repository is part of my working code (means: "unstable").

## Features

- QP-based Inverse Kinematics using [CVXOPT](http://cvxopt.org/index.html)
  (allows for task-based whole-body optimization, does not require IKFast)
- Contact double-description method for whole-body multi-contact planning
- Jacobians and hessians for the center of mass, ZMP, angular momentum, ...

## Dependencies

- [CVXOPT](http://cvxopt.org/)
  - used for Quadratic Programming
  - tested with version 1.1.7
- [OpenRAVE](https://github.com/rdiankov/openrave)
  - used for forward kinematics and visualization
  - branch: latest\_stable
  - tested with commit: f68553cb7a4532e87f14cf9db20b2becedcda624
  - you may need to [fix the Collision report issue](https://github.com/rdiankov/openrave/issues/333#issuecomment-72191884)
- [NumPy](http://www.numpy.org/)
  - tested with version 1.8.2
- [pycddlib](https://pycddlib.readthedocs.org/en/latest/)
  - used for multi-contact stability
  - installation: `pip install pycddlib`

## Installation

From the top folder, run: `sudo python setup.py install`
