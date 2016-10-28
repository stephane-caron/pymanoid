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

import matplotlib
import pylab

from scipy.spatial import ConvexHull
from numpy import array, dot, sqrt


def norm(v):
    """Euclidean norm, 2x faster than numpy.linalg.norm on my machine."""
    return sqrt(dot(v, v))


def normalize(v):
    """Normalize a vector. Don't catch ZeroDivisionError on purpose."""
    return v / norm(v)


def plot_polygon(poly, alpha=.4, color='g', linestyle='solid', fill=True,
                 linewidth=None, **kwargs):
    if type(poly) is list:
        poly = array(poly)
    ax = pylab.gca()
    hull = ConvexHull(poly)
    poly = poly[hull.vertices, :]
    xmin1, xmax1, ymin1, ymax1 = pylab.axis()
    xmin2, ymin2 = 1.5 * poly.min(axis=0)
    xmax2, ymax2 = 1.5 * poly.max(axis=0)
    pylab.axis((min(xmin1, xmin2), max(xmax1, xmax2),
                min(ymin1, ymin2), max(ymax1, ymax2)))
    patch = matplotlib.patches.Polygon(
        poly, alpha=alpha, color=color, linestyle=linestyle, fill=fill,
        linewidth=linewidth, **kwargs)
    ax.add_patch(patch)
