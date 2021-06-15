# -*- coding: utf-8 -*-
""" GCN utils.py

Created on 23-02-21

@author: Pierre-François De Plaen
"""

import scipy.sparse as sp
import numpy as np
import tensorflow as tf
from scipy.sparse.linalg.eigen.arpack import eigsh

def model_preprocessing(parameters, adj, features):
    A = sp.csr_matrix(adj.astype(float))

    features = sp.lil_matrix(features.astype(float))
    if parameters.preprocess_feats == True:
        features = preprocess_features(features) # sorte de normalization + tuple transformation

    features = sparse_to_tuple(features.astype(np.float32))
    num_features_nnz = features[1].shape
    features = tf.SparseTensor(*features)

    if parameters.model == 'gcn':
        # D^-0.5 A D^-0.5
        support = [preprocess_adj(A)]
        num_supports = 1
    elif parameters.model == 'gcn_cheby':
        support = chebyshev_polynomials(A, parameters.max_degree)
        num_supports = 1 + parameters.max_degree
    else:
        raise ValueError('Invalid argument for model: ' + str(parameters.model))

    support = [tf.cast(tf.SparseTensor(*support[0]), dtype=tf.float32)]

    feed_data = dict()
    if parameters.loss_regularization == "P":
        P = A / adj.sum(1)
        feed_data["P"] = tf.constant(P,dtype=tf.float32)
    elif parameters.loss_regularization == "L":
        feed_data["A"] = tf.constant(adj, dtype=tf.float32)
        d= adj.sum(1)
        feed_data["d"] = tf.constant(d, shape=(d.shape[0],1), dtype=tf.float32)

    return support, features, feed_data, num_features_nnz


def parse_index_file(filename):
    """
    Parse index file.
    """
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """
    Create mask.
    """
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def sparse_to_tuple(sparse_mx):
    """
    Convert sparse matrix to tuple representation.
    """
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """
    Row-normalize feature matrix
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1, where=rowsum!=0).flatten() # 1/rowsum, [2708]
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return features.astype(np.float32)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)) # D
    d_inv_sqrt = np.power(rowsum, -0.5).flatten() # D^-0.5
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt) # D^-0.5
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo() # D^-0.5AD^0.5


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)





def chebyshev_polynomials(adj, k):
    """
    Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation).
    """
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)