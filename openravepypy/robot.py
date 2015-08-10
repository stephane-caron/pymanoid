#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <caron@phare.normalesup.org>
#
# This file is part of openravepypy.
#
# openravepypy is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# openravepypy is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# openravepypy. If not, see <http://www.gnu.org/licenses/>.


import time

from numpy import arange, array, cross, dot, eye, ones, maximum
from numpy import zeros, hstack, vstack, tensordot, minimum
from openravepy import RaveCreateModule
from rotation import crossmat
from inverse_kinematics import DiffIKSolver


# Notations and names
# ===================
#
# am: Angular Momentum
# am_rate: Rate (time-derivative) of Angular Momentum
# c: link COM
# m: link mass
# omega: link angular velocity
# r: origin of link frame
# R: link rotation
# T: link transform
# v: link velocity (v = [rd, omega])
#
# Unless otherwise mentioned, coordinates are in the absolute reference frame.


class Robot(object):

    def __init__(self, env, robot_name, vel_lim=10.):
        env.GetPhysicsEngine().SetGravity(array([0, 0, -9.81]))
        rave = env.GetRobot(robot_name)
        q_min, q_max = rave.GetDOFLimits()
        rave.SetDOFVelocityLimits(1000 * rave.GetDOFVelocityLimits())

        self.active_dofs = None
        self.env = env
        self.mass = sum([link.GetMass() for link in rave.GetLinks()])
        self.nb_dof = rave.GetDOF()
        self.q_max = q_max
        self.q_min = q_min
        self.qd_max = +vel_lim * ones(self.nb_dof)
        self.qd_min = -vel_lim * ones(self.nb_dof)
        self.rave = rave

    #
    # Kinematics
    #

    def get_dof_values(self, dof_indices=None):
        if dof_indices is not None:
            return self.rave.GetDOFValues(dof_indices)
        return self.rave.GetDOFValues()

    @property
    def q(self):
        return self.get_dof_values()

    @property
    def q_active(self):
        return self.get_dof_values(self.active_dofs)

    def set_active_dof_values(self, q_active):
        self.rave.SetDOFValues(q_active, self.active_dofs)

    def set_active_dofs(self, active_dofs):
        self.active_dofs = active_dofs

    def set_dof_values(self, q, dof_indices=None):
        if dof_indices is not None:
            return self.rave.SetDOFValues(q, dof_indices)
        print "len(q) =", len(q)
        assert len(q) == self.nb_dof
        return self.rave.SetDOFValues(q)

    #
    # Inverse Kinematics
    #

    def init_ik(self, active_dofs, K_doflim, reg_weight=1e-5,
                dof_lim_scale=0.95):
        self.set_active_dofs(active_dofs)
        q_avg = .5 * (self.q_max + self.q_min)
        q_dev = .5 * (self.q_max - self.q_min)
        q_max = (q_avg + dof_lim_scale * q_dev)[active_dofs]
        q_min = (q_avg - dof_lim_scale * q_dev)[active_dofs]
        qd_max = self.qd_max[active_dofs]
        qd_min = self.qd_min[active_dofs]
        self.ik = DiffIKSolver(
            (q_min, q_max), (qd_min, qd_max), K_doflim, reg_weight,
            dof_lim_scale)

    def add_com_objective(self, target, gain, weight):
        def err(q):
            cur_com = self.compute_com(q, self.active_dofs)
            return target.p - cur_com

        def J(q):
            return self.compute_com_jacobian(q, self.active_dofs)

        self.ik.add_objective(err, J, gain, weight)

    def add_link_objective(self, link, target, gain, weight):
        def err(q):
            cur_pose = self.compute_link_pose(link, q, self.active_dofs)
            return target.pose - cur_pose

        def J(q):
            pose = self.compute_link_pose(link, q, self.active_dofs)
            J = self.compute_link_pose_jacobian(link, q, self.active_dofs)
            if pose[0] < 0:  # convention: cos(alpha) > 0
                J[:4, :] *= -1
            return J

        self.ik.add_objective(err, J, gain, weight)

    def step_tracker(self):
        q = self.q
        dq = self.ik.compute_delta(q)
        self.set_active_dof_values(q + dq)

    def solve_ik(self, dt=.1, max_it=100, conv_tol=1e-4, debug=True):
        cur_obj = 1000
        q = self.q_active
        q_max = self.ik.q_max
        q_min = self.ik.q_min
        for itnum in xrange(max_it):
            prev_obj = cur_obj
            cur_obj = self.ik.compute_objective(q)
            if abs(cur_obj - prev_obj) < conv_tol:
                break
            qd = self.ik.compute_delta(q)
            q = minimum(maximum(q_min, q + qd * dt), q_max)
            if debug:
                print "%2d:" % itnum, cur_obj
        self.set_active_dof_values(q)
        return itnum, cur_obj

    #
    # Visualization
    #

    def play_trajectory(self, traj, callback=None, dt=3e-2, dof_indices=None):
        trange = list(arange(0, traj.T, dt))
        for t in trange:
            q = traj.q(t)
            qd = traj.qd(t)
            qdd = traj.qdd(t)
            self.set_dof_values(q, dof_indices)
            if callback:
                callback(t, q, qd, qdd)
            time.sleep(dt)

    def record_trajectory(self, traj, fname='output.mpg', codec=13,
                          framerate=24, width=800, height=600, dt=3e-2,
                          show_codecs=False):
        viewer = self.env.GetViewer()
        recorder = RaveCreateModule(self.env, 'viewerrecorder')
        if show_codecs:  # linux only
            print "Available codecs:", recorder.SendCommand('GetCodecs')
        self.env.AddModule(recorder, '')
        self.rave.SetDOFValues(traj.q(0))
        recorder.SendCommand('Start %d %d %d codec %d timing '
                             'simtime filename %s\n'
                             'viewer %s' % (width, height, framerate, codec,
                                            fname, viewer.GetName()))
        time.sleep(1.)
        self.play_trajectory(traj, dt=dt)
        time.sleep(1.)
        recorder.SendCommand('Stop')
        self.env.Remove(recorder)

    def set_transparency(self, transparency):
        for link in self.rave.GetLinks():
            for geom in link.GetGeometries():
                geom.SetTransparency(transparency)

    #
    # Inertial properties
    #

    @property
    def com(self):
        return self.compute_com(self.get_dof_values())

    def compute_inertia_matrix(self, q, external_torque=None):
        M = zeros((self.nb_dof, self.nb_dof))
        self.rave.SetDOFValues(q)
        for (i, e_i) in enumerate(eye(self.nb_dof)):
            tm, _, _ = self.rave.ComputeInverseDynamics(
                e_i, external_torque, returncomponents=True)
            M[:, i] = tm
        return M

    #
    # Positions
    #

    def compute_com(self, q, dof_indices=None):
        total = zeros(3)
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                c = link.GetGlobalCOM()
                total += m * c
        return total / self.mass

    def compute_link_pos(self, link, q, link_coord=None, dof_indices=None):
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            T = link.T
            if link_coord is None:
                return T[:3, 3]
            return dot(T, hstack([link_coord, 1]))[:3]

    def compute_link_pose(self, link, q, dof_indices=None):
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            return link.pose  # first coefficient will be positive

    #
    # Velocities
    #

    def compute_angular_momentum(self, q, qd, p):
        """Compute the angular momentum with respect to point p.

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point, either a fixed point or the instantaneous COM,
        in world coordinates

        """
        momentum = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            for link in self.rave.GetLinks():
                T = link.GetTransform()
                R, r = T[0:3, 0:3], T[0:3, 3]
                c_local = link.GetLocalCOM()  # in local RF
                c = r + dot(R, c_local)

                v = link.GetVelocity()
                rd, omega = v[:3], v[3:]
                cd = rd + cross(omega, dot(R, c_local))

                m = link.GetMass()
                I = link.GetLocalInertia()  # in local RF
                momentum += cross(c - p, m * cd) \
                    + dot(R, dot(I, dot(R.T, omega)))
        return momentum

    def compute_cam(self, q, qd):
        """Compute Centroidal Angular Momentum (CAM), i.e. angular momentum at
        the instantaneous COM."""
        return self.compute_angular_momentum(q, qd, self.compute_com(q))

    def compute_com_velocity(self, q, qd):
        total = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                R = link.GetTransform()[0:3, 0:3]
                c_local = link.GetLocalCOM()
                v = link.GetVelocity()
                rd, omega = v[:3], v[3:]
                cd = rd + cross(omega, dot(R, c_local))
                total += m * cd
        return total / self.mass

    #
    # Accelerations
    #

    def compute_cam_rate(self, q, qd, qdd):
        J = self.compute_cam_pseudo_jacobian(q)
        H = self.compute_cam_pseudo_hessian(q)
        return dot(J, qdd) + dot(qd, dot(H, qd))

    def compute_com_acceleration(self, q, qd, qdd):
        J = self.compute_com_jacobian(q)
        H = self.compute_com_hessian(q)
        return dot(J, qdd) + dot(qd, dot(H, qdd))

    def compute_zmp(self, q, qd, qdd):
        global pb_times, total_times, cum_ratio, avg_ratio
        g = array([0, 0, -9.81])
        f0 = self.mass * g[2]
        tau0 = zeros(3)
        with self.rave:
            self.rave.SetDOFValues(q)
            self.rave.SetDOFVelocities(qd)
            link_velocities = self.rave.GetLinkVelocities()
            link_accelerations = self.rave.GetLinkAccelerations(qdd)
            for link in self.rave.GetLinks():
                mi = link.GetMass()
                ci = link.GetGlobalCOM()
                I_ci = link.GetLocalInertia()
                Ri = link.GetTransform()[0:3, 0:3]
                ri = dot(Ri, link.GetLocalCOM())
                # linvel = link_velocities[link.GetIndex()][:3]
                angvel = link_velocities[link.GetIndex()][3:]
                linacc = link_accelerations[link.GetIndex()][:3]
                angacc = link_accelerations[link.GetIndex()][3:]
                # ci_dot = linvel + cross(angvel, ri)
                ci_ddot = linacc \
                    + cross(angvel, cross(angvel, ri)) \
                    + cross(angacc, ri)
                angmmt = dot(I_ci, angacc) - cross(dot(I_ci, angvel), angvel)
                f0 -= mi * ci_ddot[2]
                tau0 += mi * cross(ci, g - ci_ddot) - dot(Ri, angmmt)
        return cross(array([0, 0, 1]), tau0) * 1. / f0

    #
    # Jacobians
    #

    def compute_am_as_jacobian(self, q, p):
        """Compute a matrix J(p) such that the angular momentum with respect to
        p is

            L(q, qd) = dot(J(q), qd).

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point, either a fixed point or the instantaneous COM,
        in world coordinates

        """
        J = zeros((3, len(q)))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                i = link.GetIndex()
                c = link.GetGlobalCOM()
                R = link.GetTransform()[0:3, 0:3]
                I = dot(R, dot(link.GetLocalInertia(), R.T))
                J_trans = self.rave.ComputeJacobianTranslation(i, c)
                J_rot = self.rave.ComputeJacobianAxisAngle(i)
                J += dot(crossmat(c - p), m * J_trans) + dot(I, J_rot)
        return J

    def compute_cam_as_jacobian(self, q):
        return self.compute_am_pseudo_jacobian(q, self.compute_com(q))

    def compute_com_jacobian(self, q, dof_indices=None):
        Jcom = zeros((3, self.nb_dof))
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                index = link.GetIndex()
                com = link.GetGlobalCOM()
                m = link.GetMass()
                J = self.rave.ComputeJacobianTranslation(index, com)
                Jcom += m * J
            J = Jcom / self.mass
            if dof_indices is not None:
                return J[:, dof_indices]
            return J

    def compute_link_jacobian(self, link, q, dof_indices=None):
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            J_trans = self.rave.ComputeJacobianTranslation(link.index, link.p)
            J_rot = self.rave.ComputeJacobianAxisAngle(link.index)
            J = vstack([J_rot, J_trans])
            if dof_indices is not None:
                return J[:, dof_indices]
            return J

    def compute_link_pose_jacobian(self, link, q, dof_indices=None):
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            J_trans = self.rave.CalculateJacobian(link.index, link.p)
            or_quat = link.rave.GetTransformPose()[:4]  # don't use link.pose
            J_quat = self.rave.CalculateRotationJacobian(link.index, or_quat)
            if or_quat[0] < 0:  # we enforce positive first coefficients
                J_quat *= -1.
            J = vstack([J_quat, J_trans])
            if dof_indices is not None:
                return J[:, dof_indices]
            return J

    def compute_link_translation_jacobian(self, link, q, link_coord=None,
                                          dof_indices=None):
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            p = self.compute_link_pos(link, q, link_coord, dof_indices)
            J = self.rave.ComputeJacobianTranslation(link.index, p)
            if dof_indices is not None:
                return J[:, dof_indices]
            return J

    #
    # Hessians
    #

    def compute_am_rate_as_hessian(self, q, p):
        """Returns a matrix H(q) such that the rate of change of the angular
        momentum with respect to point p is

            Ld(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd)),

        where J(q) is the result of self.compute_pseudo_jacobian(q, p).

        q -- joint angle values
        qd -- joint-angle velocities
        p -- application point, either a fixed point or the instantaneous COM,
        in world coordinates

        """
        def crosstens(M):
            assert M.shape[0] == 3
            Z = zeros(M.shape[1])
            T = array([[Z, -M[2, :], M[1, :]],
                       [M[2, :], Z, -M[0, :]],
                       [-M[1, :], M[0, :], Z]])
            return T.transpose([2, 0, 1])  # T.shape == (M.shape[1], 3, 3)

        def middot(M, T):
            """Dot product of a matrix with the mid-coordinate of a 3D tensor.

            M -- matrix with shape (n, m)
            T -- tensor with shape (a, m, b)

            Outputs a tensor of shape (a, n, b).

            """
            return tensordot(M, T, axes=(1, 1)).transpose([1, 0, 2])

        H = zeros((len(q), 3, len(q)))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                m = link.GetMass()
                i = link.GetIndex()
                c = link.GetGlobalCOM()
                R = link.GetTransform()[0:3, 0:3]
                # J_trans = self.rave.ComputeJacobianTranslation(i, c)
                J_rot = self.rave.ComputeJacobianAxisAngle(i)
                H_trans = self.rave.ComputeHessianTranslation(i, c)
                H_rot = self.rave.ComputeHessianAxisAngle(i)
                I = dot(R, dot(link.GetLocalInertia(), R.T))
                H += middot(crossmat(c - p), m * H_trans) \
                    + middot(I, H_rot) \
                    - dot(crosstens(dot(I, J_rot)), J_rot)
        return H

    def compute_cam_as_hessian(self, q):
        return self.compute_amd_pseudo_hessian(q, self.compute_com(q))

    def compute_com_hessian(self, q):
        Hcom = zeros((self.nb_dof, 3, self.nb_dof))
        with self.rave:
            self.rave.SetDOFValues(q)
            for link in self.rave.GetLinks():
                index = link.GetIndex()
                com = link.GetGlobalCOM()
                m = link.GetMass()
                H = self.rave.ComputeHessianTranslation(index, com)
                Hcom += m * H
        return Hcom / self.mass

    def compute_link_hessian(self, link, q, dof_indices=None):
        with self.rave:
            if dof_indices is not None:
                self.rave.SetDOFValues(q, dof_indices)
            else:
                self.rave.SetDOFValues(q)
            H_trans = self.rave.ComputeHessianTranslation(link.index, link.p)
            H_rot = self.rave.ComputeHessianAxisAngle(link.index)
            return hstack([H_rot, H_trans])
