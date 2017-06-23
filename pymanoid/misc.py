#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation, either version 3 of the License, or (at your option) any later
# version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with
# pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import array, dot, eye, sqrt, tensordot, zeros

from optim import solve_qp


class NDPolynomial(object):

    """
    Polynomial class with vector-valued coefficients.

    Parameters
    ----------
    coeffs : list of arrays
        Coefficients of the polynomial.
    """

    def __init__(self, coeffs):
        self.coeffs = coeffs
        self.shape = coeffs[0].shape

    @property
    def degree(self):
        return len(self.coeffs) - 1

    def __call__(self, x):
        value = zeros(self.shape)
        for coeff in reversed(self.coeffs):
            value *= x
            value += coeff
        return value


class Statistics(object):

    """
    Online estimator for statistics of a time series of scalar values.
    """

    def __init__(self):
        self.last_value = None
        self.n = 0
        self.x = 0.
        self.x2 = 0.
        self.x_max = None
        self.x_min = None

    def add(self, x):
        """
        Add a new value of the time series.

        Parameters
        ----------
        x : scalar
            New value.
        """
        self.last_value = x
        self.n += 1
        self.x += x
        self.x2 += x ** 2
        if self.x_max is None or x > self.x_max:
            self.x_max = x
        if self.x_min is None or x < self.x_min:
            self.x_min = x

    @property
    def avg(self):
        """Average of the time series."""
        if self.n < 1:
            return None
        return self.x / self.n

    @property
    def std(self):
        """Standard deviation of the time series."""
        if self.n < 1:
            return None
        elif self.n == 1:
            return 0.
        unbiased = sqrt(self.n * 1. / (self.n - 1))
        return unbiased * sqrt(self.x2 / self.n - self.avg ** 2)

    def __str__(self):
        return "%f +/- %f (max: %f, min: %f) over %d items" % (
            self.avg, self.std, self.x_max, self.x_min, self.n)

    def as_comp_times(self, unit):
        scale = {'s': 1, 'ms': 1000, 'us': 1e6}[unit]
        if self.n < 1:
            return "? %s" % unit
        elif self.n == 1:
            return "%.2f %s" % (scale * self.avg, unit)
        return "%.2f +/- %.2f %s (max: %.2f %s, min: %.2f %s) over %d items" % (
            scale * self.avg, scale * self.std, unit, scale * self.x_max, unit,
            scale * self.x_min, unit, self.n)


class PointWrap(object):

    """
    An object with a ``p`` array field.

    Parameters
    ----------
    p : list or array
        Point coordinates.
    """

    def __init__(self, p):
        assert len(p) == 3, "Argument is not a point"
        self.p = array(p)


class PoseWrap(object):

    """
    An object with a ``pose`` array field.

    Parameters
    ----------
    p : list or array
        Pose coordinates.
    """

    def __init__(self, pose):
        assert len(pose) == 7, "Argument is not a pose"
        self.pose = array(pose)


def info(msg):
    print "\033[0;32m[pymanoid] Info:\033[0;0m", msg


def is_positive_combination(b, A):
    """
    Check if b can be written as a positive combination of lines from A.

    Parameters
    ----------
    b : array
        Test vector.
    A : array
        Matrix of line vectors to combine.

    Returns
    -------
    is_positive_combination : bool
        Whether :math:`b = A^T x` for some positive `x`.

    Notes
    -----
    As an alternative implementation, one could try solving a QP minimizing
    :math:`\\|A x - b\\|^2` (and no equality constraint), however I found that
    the precision of the output is then quite low (~1e-1).
    """
    try:
        m = A.shape[0]
        P, q, G, h = eye(m), zeros(m), -eye(m), zeros(m)
        x = solve_qp(P, q, G, h, A.T, b)
        if x is None:  # optimum not found
            return False
    except ValueError:
        return False
    return norm(dot(A.T, x) - b) < 1e-10 and min(x) > -1e-10


def is_redundant(vectors):
    """
    Check if a set of vectors is redundant, i.e. one of them can be written as
    positive combination of the others.

    Parameters
    ----------
    vectors : list of arrays
        List of vectors to check.

    Note
    ----
    When using CVXOPT as QP solver, this function may print out a significant
    number of messages "Terminated (singular KKT matrix)." in the terminal.
    """
    F = array(vectors)
    all_lines = set(range(F.shape[0]))
    for i in all_lines:
        if is_positive_combination(F[i], F[list(all_lines - set([i]))]):
            return True
    return False


def middot(M, T):
    """
    Dot product of a matrix with the mid-coordinate of a 3D tensor.

    Parameters
    ----------
    M : array, shape=(n, m)
        Matrix to multiply.
    T : array, shape=(a, m, b)
        Tensor to multiply.

    Returns
    -------
    U : array, shape=(a, n, b)
        Dot product between `M` and `T`.
    """
    return tensordot(M, T, axes=(1, 1)).transpose([1, 0, 2])


def norm(v):
    """
    Euclidean norm.

    Parameters
    ----------
    v : array
        Any vector.

    Returns
    -------
    n : scalar
        Euclidean norm of `v`.

    Note
    ----
    This straightforward function is 2x faster than :func:`numpy.linalg.norm` on
    my machine.
    """
    return sqrt(dot(v, v))


def normalize(v):
    """
    Normalize a vector.

    Parameters
    ----------
    v : array
        Any vector.

    Returns
    -------
    nv : array
        Unit vector directing `v`.

    Note
    ----
    This method doesn't catch ``ZeroDivisionError`` exceptions on purpose.
    """
    return v / norm(v)


def plot_polygon(points, alpha=.4, color='g', linestyle='solid', fill=True,
                 linewidth=None):
    """
    Plot a polygon in matplotlib.

    Parameters
    ----------
    points : list of arrays
        List of poitns.
    alpha : scalar, optional
        Transparency value.
    color : string, optional
        Color in matplotlib format.
    linestyle : scalar, optional
        Line style in matplotlib format.
    fill : bool, optional
        When ``True``, fills the area inside the polygon.
    linewidth : scalar, optional
        Line width in matplotlib format.
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


def warn(msg):
    print "\033[1;33m[pymanoid] Warning:\033[0;0m", msg
