#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015 Stephane Caron <stephane.caron@normalesup.org>
#
# This file is part of pymanoid.
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


from bisect import bisect
from numpy import arange, array, poly1d, zeros


class Chunk(object):

    def __init__(self, T, q, qd, qdd=None):
        """Constructor.

        T -- duration
        q -- position function t -> q(t)
        qd -- velocity function t -> qd(t)
        qdd -- (optional) acceleration function t -> qdd(t)

        """
        self.T = T
        self.q = q
        self.qd = qd
        self.qdd = qdd

    @property
    def q_beg(self):
        return self.q(0)

    @property
    def qd_beg(self):
        return self.qd(0)

    @property
    def qdd_beg(self):
        return self.qdd(0)

    @property
    def q_end(self):
        return self.q(self.T)

    @property
    def qd_end(self):
        return self.qd(self.T)

    @property
    def qdd_end(self):
        return self.qdd(self.T)

    def retime(self, T2, s, sd, sdd):
        q2 = lambda t: self.q(s(t))
        qd2 = lambda t: sd(t) * self.qd(s(t))
        if not self.qdd:
            return Chunk(T2, q2, qd2)
        qdd2 = lambda t: sdd(t) * self.qd(s(t)) + sd(t) ** 2 * self.qdd(s(t))
        return Chunk(T2, q2, qd2, qdd2)

    def timescale(self, scaling):
        T2 = scaling * self.T
        s = poly1d([1. / scaling, 0])
        sd = s.deriv(1)
        sdd = s.deriv(2)
        return self.retime(T2, s, sd, sdd)


class LinearChunk(Chunk):

    def __init__(self, T, c1, c0):
        """Linear chunk: q(t) = c1 t + c0."""
        zz = zeros(len(c0))
        self.T = T
        self.c1 = c1
        self.c0 = c0
        self.q = lambda t: c1 * t + c0
        self.qd = lambda t: c1
        self.qdd = lambda t: zz

    @staticmethod
    def interpolate(q0, q1, T=None):
        traj = LinearChunk(1., q1 - q0, q0)
        if T is not None:
            return traj.timescale(T)
        return traj

    def timescale(self, scaling):
        T2 = scaling * self.T
        d1 = self.c1 / scaling
        d0 = self.c0
        return LinearChunk(T2, d1, d0)


class QuadraticChunk(Chunk):

    def __init__(self, T, c2, c1, c0):
        """Second-order polynomial chunk: q(t) = a t^2 + b t + c."""
        self.T = T
        self.c2 = c2
        self.c1 = c1
        self.c0 = c0
        self.q = lambda t: c2 * t ** 2 + c1 * t + c0
        self.qd = lambda t: 2 * c2 * t + c1
        self.qdd = lambda t: 2 * c2

    @staticmethod
    def interpolate(q0, q1, qd0=None, qd1=None, T=None):
        assert (qd0 is not None) != (qd1 is not None)
        if qd0 is not None:
            c2 = q1 - q0 - qd0
            c1 = qd0
            c0 = q0
        elif qd1 is not None:
            Delta_q = q1 - q0
            c2 = qd1 - Delta_q
            c1 = -qd1 + 2 * Delta_q
            c0 = q0
        traj = QuadraticChunk(1., c2, c1, c0)
        if T is not None:
            return traj.timescale(T)
        return traj

    def timescale(self, scaling):
        T2 = scaling * self.T
        d2 = self.c2 / scaling ** 2
        d1 = self.c1 / scaling
        d0 = self.c0
        return QuadraticChunk(T2, d2, d1, d0)


class CubicChunk(Chunk):

    def __init__(self, T, c3, c2, c1, c0):
        """Third-order polynomial chunk: q(t) = c3 t^3 + c2 t^2 + c1 t + c0."""
        self.T = T
        self.c3 = c3
        self.c2 = c2
        self.c1 = c1
        self.c0 = c0
        self.q = lambda t: c3 * t ** 3 + c2 * t ** 2 + c1 * t + c0
        self.qd = lambda t: 3 * c3 * t ** 2 + 2 * c2 * t + c1
        self.qdd = lambda t: 6 * c3 * t + 2 * c2

    @staticmethod
    def interpolate(q0, qd0, q1, qd1, T=None):
        """Bezier interpolation between (q0, qd0) and (q1, qd1)."""
        v1 = q0 + qd0 / 3.
        v2 = q1 - qd1 / 3.
        c3 = q1 - q0 + 3 * (v1 - v2)
        c2 = 3 * (v2 - 2 * v1 + q0)
        c1 = 3 * (v1 - q0)
        c0 = q0
        traj = CubicChunk(1., c3, c2, c1, c0)
        if T is not None:
            return traj.timescale(T)
        return traj

    def timescale(self, scaling):
        T2 = scaling * self.T
        d3 = self.c3 / scaling ** 3
        d2 = self.c2 / scaling ** 2
        d1 = self.c1 / scaling
        d0 = self.c0
        return CubicChunk(T2, d3, d2, d1, d0)


class Trajectory(object):

    def __init__(self, chunks, dof_indices=None):
        Ts = [traj.T for traj in chunks]
        if dof_indices is not None:
            assert len(dof_indices) == len(chunks[0].q(0))
        self.T = sum(Ts)
        self.chunks = chunks
        self.dof_indices = dof_indices
        self._cum_T = [sum(Ts[0:i]) for i in xrange(len(chunks) + 1)]

    @staticmethod
    def load(fname, dof_indices=None):
        assert fname[-4:] == '.pos'
        pos_list = []
        chunks = []
        with open(fname, 'r') as f:
            for line in f:
                split = line.split(' ')
                t = float(split[0])
                q = array([float(x) for x in split[1:]])
                pos_list.append((t, q))
        prev_t, prev_q = pos_list[0]
        for t, q in pos_list[1:]:
            dt = t - prev_t
            chunks.append(LinearChunk.interpolate(prev_q, q, dt))
            prev_t, prev_q = t, q
        return Trajectory(chunks, dof_indices)

    def save(self, fname, dt):
        with open(fname, 'w') as f:
            for t in arange(0, self.T + 1e-10, dt):
                f.write(('%f ' % t) + ' '.join(map(str, self.q(t))) + '\n')

    def chunk_at(self, t, return_chunk_index=False):
        i = bisect(self._cum_T, t)
        chunk_index = min(len(self.chunks), i) - 1
        t_start = self._cum_T[chunk_index]
        chunk = self.chunks[chunk_index]
        if return_chunk_index:
            return chunk, (t - t_start), chunk_index
        return chunk, (t - t_start)

    def q(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.q(t2)

    def qd(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qd(t2)

    def qdd(self, t):
        chunk, t2 = self.chunk_at(t)
        return chunk.qdd(t2)

    def retime(self, T2, s, sd, sdd):
        c = Chunk(self.T, self.q, self.qd, self.qdd)
        return Trajectory([c.retime(T2, s, sd, sdd)])

    def timescale(self, scaling):
        c = Chunk(self.T, self.q, self.qd, self.qdd)
        return Trajectory([c.timescale(scaling)])
