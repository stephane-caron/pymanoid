# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

### Added

- Example: [horizontal walking](examples/horizontal_walking.py) by linear model predictive control
- IK: can now take joint acceleration limits into account
- IK: upgraded with Levenberg-Marquardt damping
- IK: warm-start parameter to ``solve()``
- Point: new attribute ``point.pdd`` for the acceleration
- Point: new function ``point.integrate_constant_jerk()``
- Robot model gets a ``get_link()`` function
- Simulation gets ``set_camera_transform()`` function
- SwingFoot type: a polynomial swing foot interpolator
- This change log
- ZMP support areas can now take optional contact pressure limits

### Fixed

- IK: singularity fix from [Pfeiffer et al.](https://doi.org/10.1109/LRA.2018.2855265)
- Knee joint names in JVRC-1 model
- Python 3 compatibility
- Restore initial settings in IK solve()

### Changed

- Contact: ``copy()`` now takes optional ``hide`` keyword argument
- GUI: default point size is now 1 cm
- GUI: renamed ``draw_polyhedron()`` to ``draw_polytope()``
- IK: task strings now print both weight and gain coefficients
- MPC: can now pass optional arguments to the QP solver in ``solve()``
- MPC: no need to call the ``build()`` function any more
- Removed outdated [Copra](https://github.com/vsamy/copra/) wrapper
- Stance: now bind end-effector links as well
- Stance: simplified prototype of ``compute_zmp_support_area``

## [1.0.0] - 2018/10/17

### Added
- Initial release of the project. Let's take it from there.
