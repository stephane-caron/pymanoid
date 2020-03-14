#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2020 Stephane Caron <stephane.caron@normalesup.org>
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

"""
This example shows how to compute the contact wrench cone (CWC), a generalized
multi-contact friction cone. See [Caron15]_ for details.
"""

import IPython

from numpy import array

import pymanoid

from pymanoid import Stance


def print_contact(name, contact):
    print("%s:" % name)
    print("- pos = %s" % repr(contact.p))
    print("- rpy = %s" % repr(contact.rpy))
    print("- half-length = %s" % repr(contact.shape[0]))
    print("- half-width = %s" % repr(contact.shape[1]))
    print("- friction = %f" % contact.friction)
    print("")


if __name__ == "__main__":
    sim = pymanoid.Simulation(dt=0.03)
    robot = pymanoid.robots.JVRC1('JVRC-1.dae', download_if_needed=True)
    sim.set_viewer()
    sim.viewer.SetCamera([
        [0.60587192, -0.36596244,  0.70639274, -2.4904027],
        [-0.79126787, -0.36933163,  0.48732874, -1.6965636],
        [0.08254916, -0.85420468, -0.51334199,  2.79584694],
        [0.,  0.,  0.,  1.]])
    robot.set_transparency(0.25)
    stance = Stance.from_json('stances/triple.json')
    stance.bind(robot)
    robot.ik.solve()
    sim.schedule(robot.ik)
    sim.start()

    p = array([0., 0., 0.])
    CWC_O = stance.compute_wrench_inequalities(p)
    print_contact("Left foot", stance.left_foot)
    print_contact("Right foot", stance.right_foot)
    print("Contact Wrench Cone at %s:" % str(p))
    print("- has %d lines" % CWC_O.shape[0])

    if IPython.get_ipython() is None:
        IPython.embed()
