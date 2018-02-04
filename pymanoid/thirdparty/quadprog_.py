#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Copyright (C) 2015-2017 Stephane Caron <stephane.caron@lirmm.fr>
#
# This file is part of pymanoid <https://github.com/stephane-caron/pymanoid>.
#
# pymanoid is free software: you can redistribute it and/or modify it under the
# terms of the GNU Lesser General Public License as published by the Free
# Software Foundation, either version 3 of the License, or (at your option) any
# later version.
#
# pymanoid is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
# A PARTICULAR PURPOSE. See the GNU Lesser General Public License for more
# details.
#
# You should have received a copy of the GNU Lesser General Public License along
# with pymanoid. If not, see <http://www.gnu.org/licenses/>.

from numpy import hstack, vstack
from quadprog import solve_qp as _solve_qp


def solve_qp(P, q, G, h, A=None, b=None, sym_proj=False):
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\mathrm{minimize} \\ & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} \\ & G x \\leq h \\\\
            & A x = b

    using the `quadprog <https://pypi.python.org/pypi/quadprog/>`_ QP
    solver, which implements the Goldfarb-Idnani dual algorithm
    [Goldfarb83]_.

    Parameters
    ----------
    P : array, shape=(n, n)
        Symmetric quadratic-cost matrix.
    q : array, shape=(n,)
        Quadratic-cost vector.
    G : array, shape=(m, n)
        Linear inequality matrix.
    h : array, shape=(m,)
        Linear inequality vector.
    A : array, shape=(meq, n), optional
        Linear equality matrix.
    b : array, shape=(meq,), optional
        Linear equality vector.
    sym_proj : bool, optional
        Set to `True` when the `P` matrix provided is not symmetric.

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the QP, if found.

    Raises
    ------
    ValueError
        If the QP is not feasible.

    Note
    ----
    The quadprog solver assumes `P` is symmetric. If that is not the case, set
    `sym_proj=True` to project it on its symmetric part beforehand.
    """
    qp_G = .5 * (P + P.T) if sym_proj else P
    qp_a = -q
    if A is not None:
        qp_C = -vstack([A, G]).T
        qp_b = -hstack([b, h])
        meq = A.shape[0]
    else:  # no equality constraint
        qp_C = -G.T
        qp_b = -h
        meq = 0
    return _solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
