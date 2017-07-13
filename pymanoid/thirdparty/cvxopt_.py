
import cvxopt
import cvxopt.solvers

from cvxopt import matrix as cvxmat
from cvxopt.solvers import lp
from cvxopt.solvers import qp
from numpy import array

cvxopt.solvers.options['show_progress'] = False  # disable cvxopt output

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
    Solve a linear program defined by:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & c^T x \\\\
        \\mathrm{subject\\ to} & & G x \\leq h \\\\
            & & A x = b
        \\end{eqnarray}

    using the `CVXOPT
    <http://cvxopt.org/userguide/coneprog.html#linear-programming>`_ interface
    to LP solvers.

    Parameters
    ----------
    c : array, shape=(n,)
        Linear-cost vector.
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
    sol = lp(*args, solver=solver)
    if 'optimal' not in sol['status']:
        raise ValueError("LP optimum not found: %s" % sol['status'])
    return array(sol['x']).reshape((c.shape[0],))


def solve_qp(P, q, G, h, A=None, b=None, solver=None, sym_proj=False):
    """
    Solve a quadratic program defined as:

    .. math::

        \\begin{eqnarray}
        \\mathrm{minimize} & & (1/2) x^T P x + q^T x \\\\
        \\mathrm{subject\\ to} & & G x \\leq h \\\\
            & & A x = b
        \\end{eqnarray}

    using CVXOPT
    <http://cvxopt.org/userguide/coneprog.html#quadratic-programming>.

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
    solver : string, optional
        Set to 'mosek' to run MOSEK rather than CVXOPT.
    sym_proj : bool, optional
        Set to `True` when the `P` matrix provided is not symmetric.

    Returns
    -------
    x : array, shape=(n,)
        Solution to the QP, if found, otherwise ``None``.

    Note
    ----
    CVXOPT only considers the lower entries of `P`, assuming it is symmetric. If
    that is not the case, set `sym_proj=True` to project it on its symmetric
    part beforehand.
    """
    if sym_proj:
        P = .5 * (P + P.T)
    args = [cvxmat(P), cvxmat(q), cvxmat(G), cvxmat(h)]
    if A is not None:
        args.extend([cvxmat(A), cvxmat(b)])
    sol = qp(*args, solver=solver)
    if not ('optimal' in sol['status']):
        raise ValueError("QP optimum not found: %s" % sol['status'])
    return array(sol['x']).reshape((P.shape[1],))
