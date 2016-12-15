#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2016 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import array, dot, hstack, sqrt

from rotations import quat_slerp


class AvgStdEstimator(object):

    """Estimate average and standard deviation online for a scalar series."""

    def __init__(self):
        self.x = 0.
        self.x2 = 0.
        self.n = 0

    def add(self, v):
        self.x += v
        self.x2 += v ** 2
        self.n += 1

    @property
    def avg(self):
        if self.n < 1:
            return None
        return self.x / self.n

    @property
    def std(self):
        if self.n < 1:
            return None
        elif self.n == 1:
            return 0.
        unbiased = sqrt(self.n * 1. / (self.n - 1))
        return unbiased * sqrt(self.x2 / self.n - self.avg ** 2)

    def get_all(self):
        return (self.avg, self.std, self.n)


def norm(v):
    """Euclidean norm, 2x faster than numpy.linalg.norm on my machine."""
    return sqrt(dot(v, v))


def normalize(v):
    """Normalize a vector. Don't catch ZeroDivisionError on purpose."""
    return v / norm(v)


def plot_polygon(points, alpha=.4, color='g', linestyle='solid', fill=True,
                 linewidth=None):
    """
    Plot a polygon in matplotlib.

    INPUT:

    - ``points`` -- list of points
    - ``alpha`` -- (optional) transparency
    - ``color`` -- (optional) color in matplotlib format
    - ``linestyle`` -- (optional) line style in matplotlib format
    - ``fill`` -- (optional) when ``True``, fills the area inside the polygon
    - ``linewidth`` (optional) line width in matplotlib format
    """
    from matplotlib.patches import Polygon
    from pylab import axis, gca
    from scipy.spatial import ConvexHull
    if type(points) is list:
        points = array(points)
    ax = gca()
    hull = ConvexHull(points)
    points = points[hull.vertices, :]
    xmin1, xmax1, ymin1, ymax1 = axis()
    xmin2, ymin2 = 1.5 * points.min(axis=0)
    xmax2, ymax2 = 1.5 * points.max(axis=0)
    axis((min(xmin1, xmin2), max(xmax1, xmax2),
          min(ymin1, ymin2), max(ymax1, ymax2)))
    patch = Polygon(
        points, alpha=alpha, color=color, linestyle=linestyle, fill=fill,
        linewidth=linewidth)
    ax.add_patch(patch)


def interpolate_pose_linear(pose0, pose1, t):
    """
    Linear pose interpolation.

    INPUT:

    - ``pose0`` -- first pose
    - ``pose1`` -- end pose
    - ``t`` -- floating number between 0. and 1.

    OUTPUT:

    Pose linearly interpolated between ``pose0`` (with weight ``t``) and
    ``pose1`` (with weight ``1 - t``).
    """
    pos = pose0[4:] + t * (pose1[4:] - pose0[4:])
    quat = quat_slerp(pose0[:4], pose1[:4], t)
    return hstack([quat, pos])
