# -*- coding: utf-8 -*-
""" DCNN utils.py

Created on 23-02-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf
import scipy.sparse as sp
import numpy as np

def model_preprocessing(parameters, adj, features):
    if parameters.sparse_mode:
        P_series = convert_A_to_diffusion_sparse(adj, parameters.hops_num, fully_sparse=False)
        P_series = [tf.cast(tf.SparseTensor(*el), dtype=tf.float32) for el in P_series]
    else:
        P_series = convert_A_to_diffusion(adj, parameters.hops_num)
        P_series = tf.convert_to_tensor(P_series)

    features = sp.lil_matrix(features.astype(float))
    if parameters.preprocess_feats == True:
        features = preprocess_features(features)  # sorte de normalization + tuple transformation

    features = tf.convert_to_tensor(features.todense(), dtype=tf.float32) # features = tf.SparseTensor(*features)

    feed_data = dict()
    if parameters.loss_regularization == "P":
        P = adj / adj.sum(1)
        feed_data["P"] = tf.constant(P, dtype=tf.float32)
    elif parameters.loss_regularization == "L":
        feed_data["A"] = tf.constant(adj, dtype=tf.float32)
        d = adj.sum(1)
        feed_data["d"] = tf.constant(d, shape=(d.shape[0], 1), dtype=tf.float32)

    return P_series, features, feed_data, None


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
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


def convert_A_to_diffusion(A, k):
    """
    Computes [P**0, P**1*K, P**2*K, ..., P**k*K]

    :param A: 2d numpy array
    :param k: degree of series
    :return: 3d numpy array [P**0, P**1*K, P**2*K, ..., P**k*K]
    """
    assert k >= 0
    series = [np.identity(A.shape[0])]
    d = A.sum(1)
    P = A / d[:,None]
    P_pow = P.copy()
    if k > 0:
        series.append(P_pow.copy())
        for i in range(2, k+1):
            P_pow = np.dot(P, P_pow)
            series.append(P_pow.copy())
    return np.asarray(series, dtype=np.float32)


def convert_A_to_diffusion_sparse(A, k, fully_sparse=False):
    """
    Computes [P**0, P**1*K, P**2*K, ..., P**k*K] in sparse format

    :param A: 2d numpy array
    :param k: degree of series
    :return: 3d sparse representation containing [P**0, P**1*K, P**2*K, ..., P**k*K]
    """
    assert k >= 0
    series = [sp.identity(A.shape[0], format='csr')] # series = [np.identity(A.shape[0])]
    d_vertical = np.vstack(A.sum(1))
    P = A / d_vertical
    P_pow = P.copy()
    if k > 0:
        series.append(sp.csr_matrix(P_pow.astype(float))) # series.append(P_pow.copy())
        for i in range(2, k+1):
            P_pow = np.dot(P, P_pow)
            series.append(sp.csr_matrix(P_pow.astype(float))) # series.append(P_pow.copy())

    if fully_sparse: # Transform list of 2D sparse tuple representations to 3D sparse tuple representation
        items = []
        values = []
        size = (k+1, A.shape[0], A.shape[1])
        for id, el in enumerate(series):
            series[id] = sparse_to_tuple(el)
            for item, value in zip(series[id][0], series[id][1]):
                items.append([id, item[0], item[1]])
                values.append(value)

        return (np.asarray(items), np.asarray(values, dtype=np.float32), size)
    else:
        for id, el in enumerate(series):
            series[id] = sparse_to_tuple(el)

        return series

def preprocess_features(features):
    """
    Row-normalize feature matrix
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1, where=rowsum!=0).flatten() # 1/rowsum, [2708]
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return features.astype(np.float32)
