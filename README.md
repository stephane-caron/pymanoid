# pymanoid

Python library for humanoid robotics in OpenRAVE. Features:

- QP-based Inverse Kinematics solver
- Jacobians and hessians for the center of mass, the ZMP and the angular momentum
- Double-description method for multi-contact stability

Disclaimer: this repository is part of my working code (read: **unstable**).

## Dependencies

- [CVXOPT](http://cvxopt.org/)
  - used for Quadratic Programming
  - tested with version 1.1.7
- [OpenRAVE](https://github.com/rdiankov/openrave)
  - used for forward kinematics and visualization
  - tested with commit `f68553cb7a4532e87f14cf9db20b2becedcda624` in branch
    `latest_stable`
  - you may need to [fix the Collision report issue](https://github.com/rdiankov/openrave/issues/333#issuecomment-72191884)
- [NumPy](http://www.numpy.org/)
  - used for scientific computing
  - tested with version 1.8.2
- [pycddlib](https://pycddlib.readthedocs.org/en/latest/)
  - used for multi-contact stability
  - tested with version 1.0.5a1
  - installation: `pip install pycddlib`

## Installation

From the top folder, run: `sudo python setup.py install`
