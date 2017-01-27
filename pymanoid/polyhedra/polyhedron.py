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

import cdd

from numpy import array


class Polyhedron(object):

    """
    Polyhedron.

    Parameters
    ----------
    hrep : array, optional
        Halfspace-representation matrix [b | A] where the polyhedron is defined
        as `b + A x >= 0`.
    vrep : array, optional
        Vertex-representation matrix [t | V] where V is the matrix of stacked
        generators and t indicates their type: 1 for vertices and 0 for rays.
    """

    number_type = 'float'

    def __init__(self, hrep=None, vrep=None):
        assert hrep is not None or vrep is not None, \
            "Please provide either H-rep or V-rep."
        self.hrep = hrep
        self.vrep = vrep

    @property
    def hrep_pair(self):
        """
        Returns a pair (A, b) s.t. the polyhedron is defined by A * x <= b.
        """
        if self.hrep is None:
            return None
        bA = self.hrep
        return (-array(bA[:, 1:]), array(bA[:, 0]))

    def compute_hrep(self):
        """
        Returns a matrix [b | A] s.t. the polyhedron is defined by b + A x >= 0.
        """
        if self.hrep is not None:
            return self.hrep
        mat = cdd.Matrix(self.vrep, number_type=self.number_type)
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(mat)
        ineq = P.get_inequalities()
        if ineq.lin_set:
            raise NotImplementedError("polyhedron has linear generators")
        self.hrep = array(ineq)
        return self.hrep

    def compute_vrep(self):
        """
        Returns a matrix [t | V] of stacked rays and vertices (given by row
        vectors) where types t are given in the first column: 0 for ray and 1
        for vertex.
        """
        if self.vrep is not None:
            return self.vrep
        mat = cdd.Matrix(self.hrep, number_type=self.number_type)
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        gen = P.get_generators()
        if gen.lin_set:
            raise NotImplementedError("polyhedron has linear generators")
        self.vrep = array(gen)
        return self.vrep

    @property
    def rays(self):
        """
        Returns the list of rays of the polyhedron.
        """
        if self.vrep is None:
            return None
        tV = self.vrep
        return [tV[i, 1:] for i in xrange(tV.shape[0]) if abs(tV[i, 0]) < 1e-10]

    @property
    def vertices(self):
        """
        Returns the list of vertices of the polyhedron.
        """
        if self.vrep is None:
            return None
        tV = self.vrep
        return [tV[i, 1:] for i in xrange(tV.shape[0]) if tV[i, 0] == 1]
