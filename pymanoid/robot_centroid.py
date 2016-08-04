
from numpy import array, cross, dot, zeros, tensordot, ndarray
from robot_contact import ContactingRobot
from rotations import crossmat


class CentroidalRobot(ContactingRobot):

    """
    Center Of Mass
    ==============
    """

    @property
    def com(self):
        return self.compute_com()

    @property
    def comd(self):
        return self.compute_com_velocity()

    def compute_com(self):
        total = zeros(3)
        for link in self.rave.GetLinks():
            m = link.GetMass()
            c = link.GetGlobalCOM()
            total += m * c
        return total / self.mass

    def compute_com_velocity(self):
        total = zeros(3)
        for link in self.rave.GetLinks():
            m = link.GetMass()
            R = link.GetTransform()[0:3, 0:3]
            c_local = link.GetLocalCOM()
            v = link.GetVelocity()
            rd, omega = v[:3], v[3:]
            cd = rd + cross(omega, dot(R, c_local))
            total += m * cd
        return total / self.mass

    def compute_com_jacobian(self):
        """
        Compute the jacobian matrix J(q) of the position of the center of mass G
        of the robot, i.e. the velocity of the G is given by:

                pd_G(q, qd) = J(q) * qd

        q -- vector of joint angles
        qd -- vector of joint velocities
        pd_G -- velocity of the center of mass G
        """
        J_com = zeros((3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            if m < 1e-4:
                continue
            index = link.GetIndex()
            c = link.GetGlobalCOM()
            J_com += m * self.rave.ComputeJacobianTranslation(index, c)
        J_com /= self.mass
        return J_com

    def compute_com_acceleration(self, qdd):
        qd = self.qd
        J = self.compute_com_jacobian()
        H = self.compute_com_hessian()
        return dot(J, qdd) + dot(qd, dot(H, qdd))

    def compute_com_hessian(self):
        H_com = zeros((self.nb_dofs, 3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            if m < 1e-4:
                continue
            index = link.GetIndex()
            c = link.GetGlobalCOM()
            H_com += m * self.rave.ComputeHessianTranslation(index, c)
        H_com /= self.mass
        return H_com

    def add_com_task(self, target, gain=None, weight=None):
        """
        Add a COM tracking task to the IK.

        INPUT:

        - ``target`` -- coordinates or any Body object with a 'pos' attribute
        """
        if type(target) is list:
            target = array(target)
        if type(target) is ndarray:
            def residual(dt):
                return target - self.com
        elif hasattr(target, 'pos'):
            def residual(dt):
                return target.pos - self.com
        elif hasattr(target, 'p'):
            def residual(dt):
                return target.p - self.com
        else:  # COM target should be a position
            msg = "Target of type %s has no 'pos' attribute" % type(target)
            raise Exception(msg)

        jacobian = self.compute_com_jacobian
        self.ik.add_task('com', residual, jacobian, gain, weight)

    def update_com_task(self, target, gain=None, weight=None):
        if 'com' not in self.ik.gains or 'com' not in self.ik.weights:
            raise Exception("No COM task to update in robot IK")
        gain = self.ik.gains['com']
        weight = self.ik.weights['com']
        self.ik.remove_task('com')
        self.add_com_task(target, gain, weight)

    """
    Angular Momentum
    ================
    """

    def compute_angular_momentum(self, p):
        """
        Compute the angular momentum with respect to point p.

        INPUT:

        - ``p`` -- application point in world coordinates
        """
        am = zeros(3)
        for link in self.rave.GetLinks():
            # TODO: replace by newer link.GetGlobalInertia()
            T = link.GetTransform()
            m = link.GetMass()
            v = link.GetVelocity()
            c = link.GetGlobalCOM()
            R, r = T[0:3, 0:3], T[0:3, 3]
            I = dot(R, dot(link.GetLocalInertia(), R.T))
            rd, omega = v[:3], v[3:]
            cd = rd + cross(r - c, omega)
            am += cross(c - p, m * cd) + dot(I, omega)
        return am

    def compute_angular_momentum_jacobian(self, p):
        """
        Compute the jacobian matrix J(q) such that the angular momentum of the
        robot at p is given by:

            L_p(q, qd) = J(q) * qd

        INPUT:

        - ``p`` -- application point in world coordinates
        """
        J_am = zeros((3, self.nb_dofs))
        for link in self.rave.GetLinks():
            m = link.GetMass()
            i = link.GetIndex()
            c = link.GetGlobalCOM()
            R = link.GetTransform()[0:3, 0:3]
            I = dot(R, dot(link.GetLocalInertia(), R.T))
            J_trans = self.rave.ComputeJacobianTranslation(i, c)
            J_rot = self.rave.ComputeJacobianAxisAngle(i)
            J_am += dot(crossmat(c - p), m * J_trans) + dot(I, J_rot)
        return J_am

    def compute_angular_momentum_hessian(self, p):
        """
        Returns a matrix H(q) such that the rate of change of the angular
        momentum with respect to point p is

            Ld_p(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd)),

        where J(q) is the angular-momentum jacobian.

        p -- application point in world coordinates
        """
        def crosstens(M):
            assert M.shape[0] == 3
            Z = zeros(M.shape[1])
            T = array([[Z, -M[2, :], M[1, :]],
                       [M[2, :], Z, -M[0, :]],
                       [-M[1, :], M[0, :], Z]])
            return T.transpose([2, 0, 1])  # T.shape == (M.shape[1], 3, 3)

        def middot(M, T):
            """
            Dot product of a matrix with the mid-coordinate of a 3D tensor.

            M -- matrix with shape (n, m)
            T -- tensor with shape (a, m, b)

            Outputs a tensor of shape (a, n, b).
            """
            return tensordot(M, T, axes=(1, 1)).transpose([1, 0, 2])

        H = zeros((self.nb_dofs, 3, self.nb_dofs))
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

    """
    Centroidal Angular Momentum (CAM)
    =================================

    It is simply the angular momentum taken at the center of mass.
    """

    @property
    def cam(self):
        return self.compute_cam(self.q, self.qd)

    def compute_cam(self):
        """Compute the centroidal angular momentum."""
        return self.compute_angular_momentum(self.compute_com())

    def compute_cam_jacobian(self):
        """
        Compute the jacobian matrix J(q) such that the CAM is given by:

            L_G(q, qd) = J(q) * qd
        """
        return self.compute_angular_momentum_jacobian(self.compute_com())

    def compute_cam_rate(self, qdd):
        """Compute the time-derivative of the CAM. """
        qd = self.qd
        J = self.compute_cam_jacobian()
        H = self.compute_cam_hessian()
        return dot(J, qdd) + dot(qd, dot(H, qd))

    def compute_cam_hessian(self, q):
        """
        Compute the matrix H(q) such that the rate of change of the CAM is

            Ld_G(q, qd) = dot(J(q), qdd) + dot(qd.T, dot(H(q), qd))
        """
        p_G = self.compute_com(q)
        return self.compute_angular_momentum_hessian(q, p_G)

    def add_constant_cam_task(self, weight=None):
        """
        Try to keep the centroidal angular momentum constant.

        INPUT:

        ``weight`` -- task weight (optional)

        .. NOTE::

            The way this task is implemented may be surprising. Basically,
            keeping a constant CAM means d/dt(CAM) == 0, i.e.,

                d/dt (J_cam * qd) == 0
                J_cam * qdd + qd * H_cam * qd == 0

            Because the IK works at the velocity level, we approximate qdd by
            finite difference from the previous velocity (``qd`` argument to the
            residual function):

                J_cam * (qd_next - qd) / dt + qd * H_cam * qd == 0

            Finally, the task in qd_next (output velocity) is:

                J_cam * qd_next == J_cam * qd - dt * qd * H_cam * qd

            Hence, there are two occurrences of J_cam: one in the task residual,
            and the second in the task jacobian.

        """
        def residual(dt):
            qd = self.qd
            J_cam = self.compute_cam_jacobian()
            H_cam = self.compute_cam_hessian()  # computation intensive :(
            return dot(J_cam, qd) - dt * dot(qd, dot(H_cam, qd))

        jacobian = self.compute_cam_jacobian
        self.ik.add_task('cam', residual, jacobian, weight=weight,
                         unit_gain=True)

    def add_min_cam_task(self, weight=None):
        """
        Minimize the centroidal angular momentum.

        INPUT:

        ``weight`` -- task weight (optional)
        """
        def residual(dt):
            return zeros((3,))

        jacobian = self.compute_cam_jacobian
        self.ik.add_task('cam', residual, jacobian, weight=weight,
                         unit_gain=True)
