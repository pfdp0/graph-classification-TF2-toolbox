# -*- coding: utf-8 -*-
""" MLP utils.py

Created on 04-04-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf
import numpy as np
import scipy.sparse as sp

def model_preprocessing(parameters, adj, features):
    A = sp.csr_matrix(adj.astype(float))

    features = sp.lil_matrix(features.astype(float))
    if parameters.preprocess_feats == True:
        features = preprocess_features(features)  # sorte de normalization + tuple transformation

    features = sparse_to_tuple(features.astype(np.float32))
    num_features_nnz = features[1].shape
    features = tf.SparseTensor(*features)

    feed_data = dict()
    if parameters.loss_regularization == "P":
        P = A / adj.sum(1)
        feed_data["P"] = tf.constant(P, dtype=tf.float32)
    elif parameters.loss_regularization == "L":
        feed_data["A"] = tf.constant(adj, dtype=tf.float32)
        d = adj.sum(1)
        feed_data["d"] = tf.constant(d, shape=(d.shape[0], 1), dtype=tf.float32)

    return None, features, feed_data, num_features_nnz

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