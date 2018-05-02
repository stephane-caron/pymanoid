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
# You should have received a copy of the GNU Lesser General Public License
# along with pymanoid. If not, see <http://www.gnu.org/licenses/>.

from casadi import MX, nlpsol, vertcat


class NonlinearProgram(object):

    """
    Wrapper around `CasADi <https://github.com/casadi/casadi/wiki>`_ to
    formulate and solve nonlinear optimization problems.

    Parameters
    ----------
    solver : string, optional
        Solver name. Use 'ipopt' (default) for an interior point method or
        'sqpmethod' for sequential quadratic programming.
    options : dict, optional
        Dictionary of solver options.
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
    If set to yes, the algorithm assumes that the linear system that is solved
    to obtain the search direction, is solved sufficiently well. In that case,
    no residuals are computed, and the computation of the search direction is a
    little faster.
    """

    ipopt_fixed_variable_treatment = 'relax_bounds'
    """
    Default is "make_parameter", but "relax_bounds" seems to make computations
    faster and numerically stabler.
    """

    ipopt_linear_solver = 'ma27'
    """
    Linear solver used for step computations.
    """

    ipopt_max_cpu_time = 10.  # [s]
    """
    Maximum number of CPU seconds. Note that this parameter corresponds to
    processor time, not wall time. For a CPU with N cores, the latter can be as
    much as N times lower than the former.
    """

    ipopt_max_iter = 1000  # default is 3000
    """
    Maximum number of iterations.
    """

    ipopt_mu_strategy = "adaptive"  # default is "monotone"
    """
    Update strategy for barrier parameter: "monotone" (default) or "adaptive".
    """

    ipopt_nlp_lower_bound_inf = -0.9 * infty
    """
    Any bound below this value will be considered -inf, i.e. not lower bounded.
    """

    ipopt_nlp_upper_bound_inf = +0.9 * infty
    """
    Any bound above this value will be considered +inf, i.e. not upper bounded.
    """

    ipopt_print_level = 0  # default: 5
    """
    Output verbosity level between 0 and 12.
    """

    ipopt_print_time = False  # default: True
    """
    Print detailed solver computation times.
    """

    ipopt_warm_start_init_point = 'yes'  # default: 'no'
    """
    Indicates whether the optimization should use warm start initialization,
    where values of primal and dual variables are given (e.g. from a previous
    optimization of a related problem).
    """

    def __init__(self, solver='ipopt', options=None):
        if options is not None:
            for option in options:
                self.__setattr__('ipopt_%s' % option, options[option])
        self.cons_exprs = []
        self.cons_index = {}
        self.cons_lbounds = []
        self.cons_ubounds = []
        self.cost_function = 0
        self.initvals = []
        self.solver = None
        self.solver_name = solver
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

    def create_solver(self):
        """
        Create a new nonlinear solver.
        """
        problem = {
            'f': self.cost_function,
            'x': vertcat(*self.var_symbols),
            'g': vertcat(*self.cons_exprs)}
        options = {}
        if self.solver_name in ['scpgen', 'sqpmethod']:
            options.update({
                'qpsol': "qpoases",
                'qpsol_options': {"printLevel": "none"},
            })
        elif self.solver_name == 'ipopt':
            options.update({
                'expand': self.casadi_expand,
                'ipopt.fast_step_computation':
                self.ipopt_fast_step_computation,
                'ipopt.fixed_variable_treatment':
                self.ipopt_fixed_variable_treatment,
                'ipopt.linear_solver': self.ipopt_linear_solver,
                'ipopt.max_cpu_time': self.ipopt_max_cpu_time,
                'ipopt.max_iter': self.ipopt_max_iter,
                'ipopt.mu_strategy': self.ipopt_mu_strategy,
                'ipopt.nlp_lower_bound_inf': self.ipopt_nlp_lower_bound_inf,
                'ipopt.nlp_upper_bound_inf': self.ipopt_nlp_upper_bound_inf,
                'ipopt.print_level': self.ipopt_print_level,
                'ipopt.warm_start_init_point':
                self.ipopt_warm_start_init_point,
                'print_time': self.ipopt_print_time,
                'verbose': False,
                'verbose_init': False,
            })
        self.solver = nlpsol('solver', self.solver_name, problem, options)

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
        """
        Number of LP solver iterations applied by the NLP solver.
        """
        return self.solver.stats()['iter_count']

    @property
    def optimal_found(self):
        """
        `True` if and only if the solution is a local optimum.
        """
        return self.return_status == "Solve_Succeeded"

    @property
    def return_status(self):
        """
        String containing a status message from the NLP solver.
        """
        return self.solver.stats()['return_status']

    @property
    def solve_time(self):
        """
        Time (in seconds) taken to solve the problem.
        """
        return self.solver.stats()['t_wall_mainloop']
