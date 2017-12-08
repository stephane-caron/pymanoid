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

from thirdparty.cvxopt_ import solve_lp

try:
    # if available, provides more QP solvers
    # https://github.com/stephane-caron/qpsolvers
    from qpsolvers import solve_qp
except ImportError:
    from .thirdparty.cvxopt_ import solve_qp as cvxopt_solve_qp
    from .thirdparty.quadprog_ import solve_qp as quadprog_solve_qp

    def solve_qp(P, q, G, h, A=None, b=None, solver='quadprog', sym_proj=False):
        """
        Solve a Quadratic Program defined as:

        .. math::

            \\begin{eqnarray}
            \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
            \\mathrm{subject\\ to} & & G x \leq h \\\\
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
            Name of the QP solver to use (default is 'quadprog').
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
        """
        if solver == 'cvxopt':
            return cvxopt_solve_qp(P, q, G, h, A, b, sym_proj=sym_proj)
        elif solver == 'quadprog':
            return quadprog_solve_qp(P, q, G, h, A, b, sym_proj=sym_proj)
        raise Exception("QP solver '%s' not recognized" % solver)


__all__ = ['solve_lp', 'solve_qp']


try:
    from thirdparty.casadi_ import NonlinearProgram
    __all__ = ['solve_lp', 'solve_qp', 'NonlinearProgram']
except ImportError:
    pass
