# pymanoid

Python library for humanoid robotics in OpenRAVE.

Disclaimer: this repository is part of my working code (read: **unstable**).

Features:
- task-oriented numerical Inverse Kinematics solver [1]
- Jacobians and hessians for the center of mass, the ZMP and the angular momentum
- Double-description method for multi-contact stability

[1] compared with IKFast, this solver is slower but can deal with redundancy,
competing objectives and high-DOF systems.

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

## License

The source code is released under the GPLv3 license. In short, you can copy,
distribute and modify the software, provided that you track your changes/dates
in source files and distribute your modifications (along with their source code
and references to the original work) under the same license. For details,
please refer to the LICENSE file in the repository.
