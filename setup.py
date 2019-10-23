#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2019 Stephane Caron <stephane.caron@lirmm.fr>
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

from distutils.core import setup

setup(
    name='pymanoid',
    version='1.1.1',
    description="Python library for humanoid robotics using OpenRAVE",
    url="https://github.com/stephane-caron/pymanoid",
    author="Stéphane Caron",
    author_email="stephane.caron@lirmm.fr",
    license="GPL",
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering :: Mathematics'],
    packages=[
        'pymanoid',
        'pymanoid.pypoman',
        'pymanoid.pypoman.pypoman',
        'pymanoid.qpsolvers',
        'pymanoid.qpsolvers.qpsolvers',
        'pymanoid.robots']
)
