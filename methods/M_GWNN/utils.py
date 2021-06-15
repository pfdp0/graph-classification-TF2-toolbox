# -*- coding: utf-8 -*-
""" M_GWNN layers.py

Created on 20-05-21

@author: Pierre-François De Plaen
"""

import os
import sys
import math
from pathlib import Path

import tensorflow as tf
from sklearn.preprocessing import normalize
import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg

def model_preprocessing(parameters, adj, features):
    A = sp.csr_matrix(adj.astype(float))

    features = sp.lil_matrix(features.astype(float))
    if parameters.preprocess_feats == True:
        features = preprocess_features(features)  # sorte de normalization + tuple transformation

    features = sparse_to_tuple(features.astype(np.float32))
    num_features_nnz = features[1].shape
    features = tf.SparseTensor(*features)

    support = list()
    for s_value in parameters.wavelet_s:
        if parameters.sparse_mode:
            scale = wavelet_basis_sparse(A, s_value, parameters.laplacian_normalize, parameters.sparse_ness,
                                    s_value, parameters.weight_normalize, parameters.dataset)
        else:
            scale = wavelet_basis(A, s_value, parameters.laplacian_normalize, parameters.sparse_ness,
                                    parameters.sparse_threshold, parameters.weight_normalize, parameters.dataset)

        if parameters.sparse_mode:
            support.append([tf.cast(tf.SparseTensor(*scale[i]), dtype=tf.float32) for i in range(2)])
        else:
            support.append([tf.convert_to_tensor(s, dtype=tf.float32) for s in scale])

    feed_data = dict()
    if parameters.loss_regularization == "P":
        P = A / adj.sum(1)
        feed_data["P"] = tf.constant(P, dtype=tf.float32)
    elif parameters.loss_regularization == "L":
        feed_data["A"] = tf.constant(adj, dtype=tf.float32)
        d = adj.sum(1)
        feed_data["d"] = tf.constant(d, shape=(d.shape[0], 1), dtype=tf.float32)

    return support, features, feed_data, num_features_nnz

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

def wavelet_basis(adj,s,laplacian_normalize,sparse_ness,threshold,weight_normalize, dataset):
    root_path = Path(".")
    save_directory = '../data/saved/GWNN/'
    file_name = '{}_{}.npz'.format(dataset, laplacian_normalize)
    file_path = root_path / save_directory / file_name
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            data = np.load(f)
            lamb = data['lamb']
            U = data['U']
        #print(f"Loaded lamb and U from: '{file_path}'")
    else:
        L = laplacian(adj,normalized=laplacian_normalize)
        lamb, U = fourier(L)
        with open(file_path, 'wb') as f:
            np.savez_compressed(f, lamb=lamb, U=U)

        print(f"Saving lanb and U to: '{file_path}'")

    Weight = weight_wavelet(s,lamb,U)
    inverse_Weight = weight_wavelet_inverse(s,lamb,U)
    del U,lamb

    if (sparse_ness):
        Weight[Weight < threshold] = 0.0
        inverse_Weight[inverse_Weight < threshold] = 0.0

    if (weight_normalize == True):
        Weight = normalize(Weight, norm='l1', axis=1)
        inverse_Weight = normalize(inverse_Weight, norm='l1', axis=1)

    t_k = [inverse_Weight, Weight]

    return t_k



def wavelet_basis_sparse(adj,s,laplacian_normalize,sparse_ness,threshold,weight_normalize, dataset):
    [inverse_Weight, Weight] = wavelet_basis(adj, s, laplacian_normalize, sparse_ness, threshold, weight_normalize, dataset)

    Weight = sp.csr_matrix(Weight)
    inverse_Weight = sp.csr_matrix(inverse_Weight)

    return [sparse_to_tuple(inverse_Weight), sparse_to_tuple(Weight)]





def adj_matrix():
    names = [ 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format("cora", names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects = pkl.load(f, encoding='latin1')
            else:
                objects = pkl.load(f)
    graph = objects
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
    return adj

def laplacian(W, normalized=False):
    """Return the Laplacian of the weight matrix."""
    # Degree matrix.
    d = W.sum(axis=0)
    # Laplacian matrix.
    if not normalized:
        D = sp.diags(d.A.squeeze(), 0)
        L = D - W
    else:
        # d += np.spacing(np.array(0, W.dtype))
        d = 1 / np.sqrt(d)
        D = sp.diags(d.A.squeeze(), 0)
        I = sp.identity(d.size, dtype=W.dtype)
        L = I - D * W * D

    # assert np.abs(L - L.T).mean() < 1e-9
    assert type(L) is sp.csr.csr_matrix
    return L

def fourier(L, algo='eigh', k=100):
    """Return the Fourier basis, i.e. the EVD of the Laplacian."""
    # print "eigen decomposition:"
    def sort(lamb, U):
        idx = lamb.argsort()
        return lamb[idx], U[:, idx]
    # if(dataset == "pubmed"):
    #     # print "loading pubmed U"
    #     rfile = open("data/pubmed_U.pkl")
    #     lamb, U = pkl.load(rfile)
    #     rfile.close()
    # else:
    if algo is 'eig':
        lamb, U = np.linalg.eig(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigh':
        lamb, U = np.linalg.eigh(L.toarray())
        lamb, U = sort(lamb, U)
    elif algo is 'eigs':
        lamb, U = splinalg.eigs(L, k=k, which='SM')
        lamb, U = sort(lamb, U)
    elif algo is 'eigsh':
        lamb, U = splinalg.eigsh(L, k=k, which='SM')
    # print "end"
    # wfile = open("data/pubmed_U.pkl","w")
    # pkl.dump([lamb,U],wfile)
    # wfile.close()
    # print "pkl U end"
    return lamb, U

def weight_wavelet(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e,-lamb[i]*s)

    Weight = np.dot(np.dot(U,np.diag(lamb)),np.transpose(U))

    return Weight

def weight_wavelet_inverse(s,lamb,U):
    s = s
    for i in range(len(lamb)):
        lamb[i] = math.pow(math.e, lamb[i] * s)

    Weight = np.dot(np.dot(U, np.diag(lamb)), np.transpose(U))

    return Weight


def preprocess_features(features):
    """
    Row-normalize feature matrix
    """
    rowsum = np.array(features.sum(1)) # get sum of each row, [2708, 1]
    r_inv = np.power(rowsum, -1, where=rowsum!=0).flatten() # 1/rowsum, [2708]
    r_mat_inv = sp.diags(r_inv) # sparse diagonal matrix, [2708, 2708]
    features = r_mat_inv.dot(features) # D^-1:[2708, 2708]@X:[2708, 2708]
    return features.astype(np.float32)



# New

def get_s_coefs(alpha, beta, x1, x2):
    """Returns coefs of cubic spline s.t.
        s(x1) = 1, s(x2) = 1, s'(x1) = alpha/x1, s'(x2) = -beta/x2
    :returns: a3, a2, a1, a0 of a3*(x**3) + a2*(x**2) + ... + a0
    """
    A = np.array(
        [[x1**3, x1**2, x1, 1],
        [x2**3, x2**2, x2, 1],
        [3*(x1**3), 2*(x1**2), x1, 0],
        [3*(x2**3), 2*(x2**2), x2, 0]]
    )
    b = np.array([1, 1, alpha, -beta])

    return np.linalg.solve(A, b)


def G_new(V, alpha=1, beta=1, x1=1, x2=2):
    V = np.asarray(V)
    assert len(V.shape) <= 1

    coefs = get_s_coefs(alpha, beta, x1, x2)

    out = np.sum([V**3, V**2, V, 1] * coefs)
    out = np.where(V < x1, (V/x1)**alpha, out)
    out = np.where(V > x2, (x2/V)**beta, out)

    return out


def H_new(X, alpha=1, beta=1, x1=1, x2=2, lambda_min=0.1):
    X = np.asarray(X)
    assert len(X.shape) <= 1

    a,b,c,d = get_s_coefs(alpha, beta, x1, x2)

    x_gamma1 = (1/(3*a)) * (-b - math.sqrt(b**2 - 3*a*c))
    if 1 < x_gamma1 < 2:
        x_gamma = x_gamma1
    else:
        x_gamma = (1/(3*a)) * (-b + math.sqrt(b**2 - 3*a*c))

    gamma = G_new(x_gamma, alpha, beta, x1, x2)

    return gamma * np.exp(-np.power(X/(0.6*lambda_min), 4))