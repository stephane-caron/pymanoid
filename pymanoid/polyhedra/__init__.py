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

import polygon

from cone import Cone
from misc import is_positive_combination
from misc import is_redundant
from polygon import intersect_polygons
from polyhedron import Polyhedron
from polytope import Polytope
from projection import project_polyhedron
from projection import project_polytope

__all__ = [
    'Cone',
    'Polyhedron',
    'Polytope',
    'intersect_polygons',
    'is_positive_combination',
    'is_redundant',
    'polygon',
    'project_polyhedron',
    'project_polytope',
]
