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

import cdd

from numpy import array


class Polyhedron(object):

    """
    Wrapper for the CDD library.

    Computes both the H-rep (halfspace repsentation) and V-rep (vertex/rays
    representation) of a polyhedron, in matrix form:

    - H-rep: [b | A] where the polyhedron is defined by b + A * x >= 0
    - V-rep: [t | V] where V stacks vertices as row vectors and t is the type
                     vector (1 for points, 0 for rays/lines)

    See also: <https://github.com/haudren/pyparma>
    """

    number_type = 'float'

    def __init__(self, hrep=None, vrep=None):
        assert hrep is not None or vrep is not None, \
            "Please provide either H-rep or V-rep."
        self.__hrep = hrep
        self.__vrep = vrep

    def hrep(self):
        if self.__hrep is not None:
            return self.__hrep
        mat = cdd.Matrix(self.vrep, number_type=self.number_type)
        mat.rep_type = cdd.RepType.GENERATOR
        P = cdd.Polyhedron(mat)
        ineq = P.get_inequalities()
        if ineq.lin_set:
            raise NotImplementedError("polyhedron has linear generators")
        self.__hrep = array(ineq)
        return self.__hrep

    def vrep(self):
        if self.__vrep is not None:
            return self.__vrep
        mat = cdd.Matrix(self.hrep, number_type=self.number_type)
        mat.rep_type = cdd.RepType.INEQUALITY
        P = cdd.Polyhedron(mat)
        gen = P.get_generators()
        if gen.lin_set:
            raise NotImplementedError("polyhedron has linear generators")
        self.__vrep = array(gen)
        return self.__vrep
