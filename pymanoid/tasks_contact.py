from contact import Contact
from numpy import array, dot, ndarray
from inverse_kinematics import Task


_oppose_quat = array([-1., -1., -1., -1., +1., +1., +1.])


class LinkPoseTask(Task):

    task_type = 'link_pose'

    def __init__(self, robot, link, target, **kwargs):
        if type(target) is Contact:  # used for ROS communications
            target.robot_link = link.index  # dirty
        elif type(target) is list:
            target = array(target)

        def _pos_residual(target_pose):
            residual = target_pose - link.pose
            if dot(residual[0:4], residual[0:4]) > 1.:
                return _oppose_quat * target_pose - link.pose
            return residual

        if hasattr(target, 'pose'):
            def pos_residual():
                return _pos_residual(target.pose)
        elif type(target) is ndarray:
            def pos_residual():
                return _pos_residual(target)
        else:  # link frame target should be a pose
            raise Exception("Target %s has no 'pose' attribute" % type(target))

        def jacobian():
            return robot.compute_link_pose_jacobian(link)

        self.link = link
        Task.__init__(self, jacobian, pos_residual=pos_residual, **kwargs)

    @property
    def name(self):
        return self.link.name


class LinkPosTask(Task):

    task_type = 'link_pos'

    def __init__(self, robot, link, target, **kwargs):
        if type(target) is list:
            target = array(target)

        if hasattr(target, 'p'):
            def pos_residual():
                return target.p - link.p
        elif type(target) is ndarray:
            def pos_residual():
                return target - link.p
        else:  # this is an aesthetic comment
            raise Exception("Target %s has no 'p' attribute" % type(target))

        def jacobian():
            return robot.compute_link_pos_jacobian(link)

        self.link = link
        Task.__init__(self, jacobian, pos_residual=pos_residual, **kwargs)

    @property
    def name(self):
        return self.link.name


class ContactTask(LinkPoseTask):

    task_type = 'contact'

    def __init__(self, robot, link, target, **kwargs):
        if type(target) is Contact:  # used for ROS communications
            target.robot_link = link.index  # dirty
        elif type(target) is list:
            target = array(target)

        def pos_residual():
            residual = target.effector_pose - link.pose
            if dot(residual[0:4], residual[0:4]) > 1.:
                return _oppose_quat * target.effector_pose - link.pose
            return residual

        def jacobian():
            return robot.compute_link_pose_jacobian(link)

        self.link = link  # used by LinkPoseTask.name
        Task.__init__(self, jacobian, pos_residual=pos_residual, **kwargs)
