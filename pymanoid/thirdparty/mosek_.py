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

import cvxopt.msk
import mosek

from cvxopt_ import solve_qp as cvxopt_solve_qp

cvxopt.solvers.options['mosek'] = {mosek.iparam.log: 0}


def solve_qp(P, q, G, h, A=None, b=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} & & G x \\leq h \\\\
            & & A x = b
        \\end{eqnarray}

    Parameters
    ----------
    P : array, shape=(n, n)
        Primal quadratic cost matrix.
    q : array, shape=(n,)
        Primal quadratic cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    solver : string, optional
        Name of the QP solver to use (default is quadprog).

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the QP, if found.

    Raises
    ------
    ValueError
        If the QP is not feasible.
    """
    return cvxopt_solve_qp(P, q, G, h, A, b, 'mosek', initvals)
