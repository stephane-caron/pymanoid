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

import cvxopt
import cvxopt.solvers

from cvxopt import matrix as cvxmat
from cvxopt.solvers import lp as cvxopt_lp
from cvxopt.solvers import qp as cvxopt_qp
from numpy import array, eye, hstack, ones, vstack, zeros

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output


"""
Linear Programming
==================
"""

try:
    import cvxopt.glpk
    GLPK_IF_AVAILABLE = 'glpk'
    # GLPK is the fastest LP solver I could find so far:
    # <https://scaron.info/blog/linear-programming-in-python-with-cvxopt.html>
    # ... however, it's verbose by default, so tell it to STFU:
    cvxopt.solvers.options['glpk'] = {'msg_lev': 'GLP_MSG_OFF'}  # cvxopt 1.1.8
    cvxopt.solvers.options['msg_lev'] = 'GLP_MSG_OFF'  # cvxopt 1.1.7
    cvxopt.solvers.options['LPX_K_MSGLEV'] = 0  # previous versions
except ImportError:
    # issue a warning as GLPK is the best LP solver in practice
    print "\033[1;33m[pymanoid] Warning: GLPK solver not found\033[0;0m"
    GLPK_IF_AVAILABLE = None


def solve_lp(c, G, h, A=None, b=None, solver=GLPK_IF_AVAILABLE):
    """
    Solve a Linear Program defined by:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & c^T x \\\\
        \\mathrm{subject\\ to} & & G x \leq h \\\\
            & & A x = b
        \\end{eqnarray}

    using the `CVXOPT
    <http://cvxopt.org/userguide/coneprog.html#linear-programming>`_ interface
    to LP solvers.

    Parameters
    ----------
    c : array, shape=(n,)
        Cost vector.
    G : array, shape=(m, n)
        Linear inequality constraint matrix.
    h : array, shape=(m,)
        Linear inequality constraint vector.
    A : array, shape=(meq, n), optional
        Linear equality constraint matrix.
    b : array, shape=(meq,), optional
        Linear equality constraint vector.
    solver : string, optional
        Solver to use, default is GLPK if available

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the LP, if found, otherwise ``None``.

    Raises
    ------
    ValueError
        If the LP is not feasible.
    """
    args = [cvxmat(c), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = cvxopt_lp(*args, solver=solver)
    if 'optimal' not in sol['status']:
        raise ValueError("LP optimum not found: %s" % sol['status'])
    return array(sol['x']).reshape((c.shape[0],))


"""
Quadratic Programming
=====================
"""

try:
    from quadprog import solve_qp as _quadprog_solve_qp

    QUADPROG_IF_AVAILABLE = 'quadprog'

    def quadprog_solve_qp(P, q, G, h, A=None, b=None):
        """
        Solve a Quadratic Program defined as:

        .. math::

            \\begin{eqnarray}
            \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
            \\mathrm{subject\\ to} & & G x \leq h \\\\
                & & A x = b
            \\end{eqnarray}

        using the `quadprog <https://pypi.python.org/pypi/quadprog/>`_ QP
        solver, which implements the Goldfarb-Idnani dual algorithm [GI83]_.

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

        Returns
        -------
        x : array, shape=(n,)
            Optimal solution to the QP, if found.

        Raises
        ------
        ValueError
            If the QP is not feasible.
        """
        qp_G = .5 * (P + P.T)   # quadprog assumes that P is symmetric
        qp_a = -q
        if A is not None:
            qp_C = -vstack([A, G]).T
            qp_b = -hstack([b, h])
            meq = A.shape[0]
        else:  # no equality constraint
            qp_C = -G.T
            qp_b = -h
            meq = 0
        return _quadprog_solve_qp(qp_G, qp_a, qp_C, qp_b, meq)[0]
except ImportError:
    # issue a warning as quadprog is the best QP solver in practice
    print "\033[1;33m[pymanoid] Warning: quadprog not found\033[0;0m"
    QUADPROG_IF_AVAILABLE = None


def cvxopt_solve_qp(P, q, G, h, A=None, b=None, solver=None, initvals=None):
    """
    Solve a Quadratic Program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} & & G x \leq h \\\\
            & & A x = b
        \\end{eqnarray}

    using CVXOPT
    <http://cvxopt.org/userguide/coneprog.html#quadratic-programming>.

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
        Set to 'mosek' to run MOSEK rather than CVXOPT.
    initvals : array, shape=(n,), optional
        Warm-start guess vector.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.
    """
    # CVXOPT only considers the lower entries of P so we project on its
    # symmetric part beforehand
    P = .5 * (P + P.T)
    args = [cvxmat(P), cvxmat(q), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = cvxopt_qp(*args, solver=solver, initvals=initvals)
    if not ('optimal' in sol['status']):
        raise ValueError("QP optimum not found: %s" % sol['status'])
    return array(sol['x']).reshape((P.shape[1],))


try:
    import cvxopt.msk
    import mosek

    MOSEK_IF_AVAILABLE = 'mosek'
    cvxopt.solvers.options['mosek'] = {mosek.iparam.log: 0}

    def mosek_solve_qp(P, q, G, h, A=None, b=None, initvals=None):
        return cvxopt_solve_qp(P, q, G, h, A, b, 'mosek', initvals)
except ImportError:
    MOSEK_IF_AVAILABLE = None


def solve_qp(P, q, G, h, A=None, b=None, solver=QUADPROG_IF_AVAILABLE):
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
    if solver == 'quadprog':
        return quadprog_solve_qp(P, q, G, h, A, b)
    elif solver == 'cvxopt':
        return cvxopt_solve_qp(P, q, G, h, A, b)
    elif solver == 'mosek':
        return mosek_solve_qp(P, q, G, h, A, b)
    return cvxopt_solve_qp(P, q, G, h, A, b)


def solve_safer_qp(P, q, G, h, w_reg, w_lin, solver=MOSEK_IF_AVAILABLE):
    """
    Solve the relaxed Quadratic Program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + (1/2) \\epsilon \\| s \\|^2 q^T x
        - w 1^T s\\\\
        \\mathrm{subject\\ to} & & G x + s \\leq h \\\\
            & & s \\geq 0
        \\end{eqnarray}

    Slack variables `s` are increased by an additional term in the cost
    function, so that the solution of this "safer" QP is further inside the
    constraint region.

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
    w_reg : scalar
        Regularization term :math:`(1/2) \\epsilon` in the cost function. Set
        this parameter as small as possible (e.g. 1e-8), and increase it in case
        of numerical instability.
    w_lin : scalar
        Weight of the linear cost on slack variables. Higher values bring the
        solution further inside the constraint region but override the
        minimization of the original objective.
    solver : string, optional
        Name of the QP solver to use (default is MOSEK).

    Returns
    -------
    x : array, shape=(n,)
        Optimal solution to the relaxed QP, if found.

    Raises
    ------
    ValueError
        If the QP is not feasible.

    Notes
    -----
    The default solver for this method is MOSEK, as it seems to perform better
    on such relaxed problems.
    """
    n, m = P.shape[0], G.shape[0]
    E, Z = eye(m), zeros((m, n))
    P2 = vstack([hstack([P, Z.T]), hstack([Z, w_reg * eye(m)])])
    q2 = hstack([q, -w_lin * ones(m)])
    G2 = hstack([Z, E])
    h2 = zeros(m)
    A2 = hstack([G, -E])
    b2 = h
    return solve_qp(P2, q2, G2, h2, A2, b2, solver=solver)[:n]

"""
Nonlinear Programming
=====================
"""

try:
    from casadi import MX, nlpsol, vertcat
    CASADI_FOUND = True
except ImportError:
    CASADI_FOUND = False


class NonlinearProgram(object):

    """
    Wrapper around `CasADi <http://casadi.org>`_ to formulate and solve
    nonlinear optimization problems.
    """

    infty = 1e10

    casadi_expand = True
    """
    Replace MX with SX expressions in problem formulation.

    Note
    ----
    Setting this option to ``True`` seems to significantly improve computation
    times when using IPOPT.
    """

    """
    IPOPT options
    =============
    """

    ipopt_fast_step_computation = 'yes'  # default: 'no'
    """
    Indicates if the linear system should be solved quickly.

    If set to yes, the algorithm assumes that the linear system that is solved
    to obtain the search direction, is solved sufficiently well. In that case,
    no residuals are computed, and the computation of the search direction is a
    little faster.

    Possible values:
        - no     [Verify solution of linear system by computing residuals.]
        - yes    [Trust that linear systems are solved well.]
    """

    ipopt_fixed_variable_treatment = 'relax_bounds'
    """
    Default is "make_parameter", but "relax_bounds" seems to make computations
    faster and numerically stabler
    """

    ipopt_linear_solver = 'ma27'
    """
    Linear solver used for step computations.
    """

    ipopt_max_cpu_time = 0.1  # [s]
    """
    Maximum number of CPU seconds.

    Note
    ----
    This parameter corresponds to processor time, not wall time. For a CPU with
    N cores, the former can scale as much as N times the latter.
    """

    ipopt_max_iter = 100
    """
    Maximum number of iterations.
    """

    ipopt_mu_strategy = "adaptive"  # default is "monotone"
    """
    Update strategy for barrier parameter.

    Options are "monotone" (default) and "adaptive".

    Note
    ----
    The adaptive strategy yields ~40% faster results in practice on the COP and
    ZMP controller, while it seems to have no effect on computation times of the
    (slower) Wrench controller.
    """

    ipopt_nlp_lower_bound_inf = -0.9 * infty
    """
    Any bound less or equal this value will be considered -inf (i.e. not lower
    bounded).
    """

    ipopt_nlp_upper_bound_inf = +0.9 * infty
    """
    Any bound greater or this value will be considered +inf (i.e. not upper
    bounded).
    """

    ipopt_print_level = 0  # default: 5
    """
    Output verbosity level.

    Integer between 0 and 12.
    """

    ipopt_warm_start_init_point = 'yes'  # default: 'no'
    """
    Indicates whether this optimization should use a warm start initialization,
    where values of primal and dual variables are given (e.g. from a previous
    optimization of a related problem).
    """

    def __init__(self):
        assert CASADI_FOUND, "CasADi not found"
        self.cons_exprs = []
        self.cons_index = {}
        self.cons_lbounds = []
        self.cons_ubounds = []
        self.cost_function = 0
        self.initvals = []
        self.solver = None
        self.var_index = {}
        self.var_lbounds = []
        self.var_symbols = []
        self.var_ubounds = []

    def extend_cost(self, expr):
        """
        Add a new expression term to the cost function of the problem.

        Parameters
        ----------
        expr : casadi.MX
            Python or CasADi symbolic expression.
        """
        self.cost_function = self.cost_function + expr

    def warm_start(self, initvals):
        """
        Warm-start the problem with a new vector of initial values.

        Parameters
        ----------
        initvals : array
            Vector of initial values for all problem variables.
        """
        assert len(self.initvals) == len(initvals)
        self.initvals = list(initvals)

    def new_variable(self, name, dim, init, lb, ub):
        """
        Add a new variable to the problem.

        Parameters
        ----------
        name : string
            Variable name.
        dim : int
            Number of dimensions.
        init : array, shape=(dim,)
            Initial values.
        lb : array, shape=(dim,)
            Vector of lower bounds on variable values.
        ub : array, shape=(dim,)
            Vector of upper bounds on variable values.
        """
        assert len(init) == len(lb) == len(ub) == dim
        var = MX.sym(name, dim)
        self.var_symbols.append(var)
        self.initvals += init
        self.var_index[name] = len(self.var_lbounds)
        self.var_lbounds += lb
        self.var_ubounds += ub
        return var

    def new_constant(self, name, dim, value):
        """
        Add a new constant to the problem.

        Parameters
        ----------
        name : string
            Name of the constant.
        dim : int
            Number of dimensions.
        value : array, shape=(dim,)
            Value of the constant.

        Note
        ----
        Constants are implemented as variables with matching lower and upper
        bounds.
        """
        return self.new_variable(name, dim, init=value, lb=value, ub=value)

    def update_constant(self, name, value):
        """
        Update the value of an existing constant.

        Parameters
        ----------
        name : string
            Name of the constant.
        value : array, shape=(dim,)
            Vector of new values for the constant.
        """
        i = self.var_index[name]
        for j, v in enumerate(value):
            self.initvals[i + j] = v
            self.var_lbounds[i + j] = v
            self.var_ubounds[i + j] = v

    def update_variable_bounds(self, name, lb, ub):
        """
        Update the lower- and upper-bounds on an existing variable.

        Parameters
        ----------
        name : string
            Name of the variable.
        lb : array, shape=(dim,)
            Vector of lower bounds on variable values.
        ub : array, shape=(dim,)
            Vector of upper bounds on variable values.
        """
        assert len(lb) == len(ub)
        i = self.var_index[name]
        for j in xrange(len(lb)):
            self.var_lbounds[i + j] = lb[j]
            self.var_ubounds[i + j] = ub[j]

    def add_constraint(self, expr, lb, ub, name=None):
        """
        Add a new constraint to the problem.

        Parameters
        ----------
        expr : casadi.MX
            Python or CasADi symbolic expression.
        lb : array
            Lower-bound on the expression.
        ub : array
            Upper-bound on the expression.
        name : string, optional
            If provided, will stored the expression under this name for future
            updates.
        """
        self.cons_exprs.append(expr)
        if name is not None:
            self.cons_index[name] = len(self.cons_lbounds)
        self.cons_lbounds += lb
        self.cons_ubounds += ub

    def add_equality_constraint(self, expr1, expr2, name=None):
        """
        Add an equality constraint between two expressions.

        Parameters
        ----------
        expr1 : casadi.MX
            Expression on problem variables.
        expr2 : casadi.MX
            Expression on problem variables.
        name : string, optional
            If provided, will stored the expression under this name for future
            updates.
        """
        diff = expr1 - expr2
        dim = diff.shape[0]
        assert diff.shape[1] == 1
        zeros = [0.] * dim
        self.add_constraint(diff, zeros, zeros, name)

    def has_constraint(self, name):
        """
        Check if a given name identifies a problem constraint.
        """
        return name in self.cons_index

    def update_constraint_bounds(self, name, lb, ub):
        """
        Update lower- and upper-bounds on an existing constraint.

        Parameters
        ----------
        name : string
            Identifier of the constraint.
        lb : array
            New lower-bound of the constraint.
        ub : array
            New upper-bound of the constraint.
        """
        i = self.cons_index[name]
        for j in xrange(len(lb)):
            self.cons_lbounds[i + j] = lb[j]
            self.cons_ubounds[i + j] = ub[j]

    def create_solver(self, solver='ipopt'):
        """
        Create a new nonlinear solver.

        Parameters
        ----------
        solver : string
            Solver name. Use 'ipopt' for an interior point method or 'sqpmethod'
            for sequential quadratic programming.
        """
        problem = {
            'f': self.cost_function,
            'x': vertcat(*self.var_symbols),
            'g': vertcat(*self.cons_exprs)}
        options = {}
        if solver in ['scpgen', 'sqpmethod']:
            options.update({
                'qpsol': "qpoases",
                'qpsol_options': {"printLevel": "none"},
            })
        elif solver == 'ipopt':
            options.update({
                'expand': self.casadi_expand,
                'ipopt.fast_step_computation': self.ipopt_fast_step_computation,
                'ipopt.fixed_variable_treatment':
                self.ipopt_fixed_variable_treatment,
                'ipopt.linear_solver': self.ipopt_linear_solver,
                'ipopt.max_cpu_time': self.ipopt_max_cpu_time,
                'ipopt.max_iter': self.ipopt_max_iter,
                'ipopt.mu_strategy': self.ipopt_mu_strategy,
                'ipopt.nlp_lower_bound_inf': self.ipopt_nlp_lower_bound_inf,
                'ipopt.nlp_upper_bound_inf': self.ipopt_nlp_upper_bound_inf,
                'ipopt.print_level': self.ipopt_print_level,
                'ipopt.warm_start_init_point': self.ipopt_warm_start_init_point,
                'verbose': False
            })
        self.solver = nlpsol('solver', solver, problem, options)

    def solve(self):
        """
        Call the nonlinear solver.

        Returns
        -------
        x : array
            Vector of variable coordinates for the best solution found.
        """
        self.res = self.solver(
            x0=self.initvals, lbx=self.var_lbounds, ubx=self.var_ubounds,
            lbg=self.cons_lbounds, ubg=self.cons_ubounds)
        return self.res['x'].full().flatten()

    @property
    def iter_count(self):
        return self.solver.stats()['iter_count']

    @property
    def optimal_found(self):
        return self.return_status == "Solve_Succeeded"

    @property
    def return_status(self):
        return self.solver.stats()['return_status']

    @property
    def solve_time(self):
        return self.solver.stats()['t_wall_mainloop']
