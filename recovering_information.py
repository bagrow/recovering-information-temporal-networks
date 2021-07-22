#!/usr/bin/env python
# -*- coding: utf-8 -*-

# recovering_information.py
# Jim Bagrow
# Last Modified: 2021-07-22

import numpy as np
import numpy.linalg as LA

from sklearn import linear_model
from sklearn.linear_model import LassoLarsIC
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


@ignore_warnings(category=ConvergenceWarning)
def sparse_recover(Bmatrix,Nmatrix, alpha, max_iter=1000, tol=0.0001):
    """Sparse recovery implementation.

    Bmatrix - Graph's (unoriented) incidence matrix (numpy array)
    Nmatrix - n x T matrix (numpy array) of node time series'
    alpha - regularization hyperparameter (scalar)

    Be careful about the order of rows and columns. The rows and columns in
    Bmatrix should correspond to the rows of N and rows of (recovered) E.

    Getting Bmatrix from a networkx Graph:

    >>> import networkx as nx
    >>> G = load_network_somehow()
    >>> B = nx.incidence_matrix(G, edgelist=edgelist).todense()

    where `G` is the networkx Graph and `edgelist` is an optional list of edges
    controlling the order of columns in B.
    """

    model = linear_model.Lasso(alpha=alpha,
                               positive=True, precompute=True,
                               fit_intercept=False,
                               selection='random',
                               max_iter=max_iter, tol=tol)
    model.fit(Bmatrix, Nmatrix)
    Ereco = model.coef_.T
    return Ereco


def leastnorm_recover(Bmatrix,Nmatrix):
    """Least-norm recovery using pseudoinverse."""
    Bs = LA.pinv(Bmatrix.T).T
    return np.asarray(Bs @ Nmatrix)


@ignore_warnings(category=ConvergenceWarning)
def get_alpha_bic(Bmatrix, Nmatrix):
    """Estimate regularization hyperparameter using BIC (may be slow)."""
    Bbig,Nvec = _unroll_mats(Bmatrix, Nmatrix)
    model_bic = LassoLarsIC(criterion='bic', fit_intercept=False,
                            positive=True,eps=1e-8)
    model_bic.fit(Bbig, Nvec)
    return model_bic.alpha_


def _unroll_mats(Bmat,Nmat):
    """Stack Nmat into a vector, replicate Bmat to match."""
    r = Nmat.shape[1]
    Bbig = np.vstack([Bmat]*r)
    Nvec = np.hstack([Nmat[:,j] for j in range(r) ])
    return Bbig, Nvec
