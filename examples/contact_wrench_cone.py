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

"""
This example shows how to compute the Contact Wrench Cone, a generalized
multi-contact friction cone. See <https://scaron.info/research/rss-2015.html>
for details.
"""

import IPython

import pymanoid

from pymanoid import PointMass, Stance
from pymanoid.contact import Contact


def print_contact(name, contact):
    print "%s:" % name
    print "- pos = %s" % repr(contact.p)
    print "- rpy = %s" % repr(contact.rpy)
    print "- half-length =", contact.shape[0]
    print "- half-width =", contact.shape[1]
    print "- friction =", contact.friction
    print ""


def print_contact_wrench_cone(p=[0., 0., 0.], prec=2):
    print_contact("Left foot", stance.left_foot)
    print_contact("Right foot", stance.right_foot)
    CWC_O = stance.compute_wrench_inequalities(p)
    print "Contact Wrench Cone at %s:\n" % str(p)
    print CWC_O.round(prec)


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
    stance = Stance(
        com=PointMass(pos=[0., 0., 0.9], mass=robot.mass),
        left_foot=Contact(
            shape=robot.sole_shape,
            pos=[0.20, 0.15, 0.1],
            rpy=[0.4, 0, 0],
            friction=0.5),
        right_foot=Contact(
            shape=robot.sole_shape,
            pos=[-0.2, -0.195, 0.],
            rpy=[-0.4, 0, 0],
            friction=0.5))
    stance.bind(robot)
    robot.ik.solve()
    sim.schedule(robot.ik)
    sim.start()

    print_contact_wrench_cone()

    if IPython.get_ipython() is None:
        IPython.embed()
