# -*- coding: utf-8 -*-
""" M_GWNN layers.py

Created on 20-05-21

@author: Pierre-François De Plaen
    - inspired from https://github.com/Eilene/GWNN/tree/master/GraphWaveletNetwork
"""

from methods.GWNN.inits import *
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.python.ops.linalg.sparse import sparse_csr_matrix_ops

def sparse_dropout(x, rate, noise_shape):
    """
    Dropout for sparse tensors.
    """
    random_tensor = 1 - rate
    random_tensor += tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1./(1 - rate))


def dot(x, y, sparse=False):
    """
    Wrapper for tf.matmul (sparse vs dense).
    """
    if sparse:
        res = tf.sparse.sparse_dense_matmul(x, y)
    else:
        res = tf.linalg.matmul(x, y)
    return res


class Multi_Wavelet_Convolution(layers.Layer):
    """Multiscale Wavelet convolution layer."""
    def __init__(self, node_num, num_scales,
                 weight_normalize, input_dim, output_dim, dropout=0.,
                 is_first_layer=False,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, num_features_nonzero=None, sparse_mode=True, **kwargs):
        super(Multi_Wavelet_Convolution, self).__init__(**kwargs)

        self.node_num = node_num
        self.num_scales = num_scales
        self.weight_normalize = weight_normalize
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero
        self.dropout = dropout
        self.sparse_mode = sparse_mode

        self.is_first_layer = is_first_layer

        self.weights_ = []
        for i in range(num_scales):
             w = self.add_weight('weights_' + str(i), [input_dim, output_dim])
             self.weights_.append(w)

        # diag filter kernels (stored in a Nx1 tensor for efficient training and memory)
        self.kernels_ = []
        for i in range(num_scales):
            self.kernels_.append(self.add_weight('kernel_' + str(i), [self.node_num,1], initializer=tf.ones_initializer()))

        if self.bias:
            self.bias = self.add_weight('bias', [output_dim])

    def call(self, inputs, training=True):
        x, phis_ = inputs

        # dropout
        if self.dropout != 0:
            if training is not False and self.sparse_inputs:
                if self.is_first_layer:
                    x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
                else:
                    x = [sparse_dropout(x[i], 1 - self.dropout, self.num_features_nonzero) for i in range(len(x))]
            elif training is not False:
                if self.is_first_layer:
                    x = tf.nn.dropout(x, 1-self.dropout)
                else:
                    x = [tf.nn.dropout(x[i], 1-self.dropout) for i in range(len(x))]

        # convolve
        Z = list()
        for i, (phi_inv, phi) in enumerate(phis_):
            if self.is_first_layer:
                X_prime = dot(x, self.weights_[i], sparse=self.sparse_inputs)
            else:
                X_prime = dot(x[i], self.weights_[i], sparse=self.sparse_inputs)

            X_prime_transform = dot(phi_inv, X_prime, sparse=self.sparse_mode)
            Z.append(dot(phi, self.kernels_[i] * X_prime_transform, sparse=self.sparse_mode))


        return self.act(Z)


class Multi_scale_add_n(layers.Layer):
    def __init__(self, node_num, num_scales,
                 weight_normalize, output_dim, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, num_features_nonzero=None, sparse_mode=True, **kwargs):
        super(Multi_scale_add_n, self).__init__(**kwargs)

        self.node_num = node_num
        self.num_scales = num_scales
        self.weight_normalize = weight_normalize
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero
        self.dropout = dropout
        self.sparse_mode = sparse_mode

        self.weights_ = []
        for i in range(num_scales):
            w = self.add_weight('combination_weights_'+str(i), [1], initializer=tf.zeros_initializer())
            self.weights_.append(w)

        if self.bias:
            self.bias = self.add_weight('bias', [output_dim])

    def call(self, inputs, training=True):
        x, _ = inputs

        # dropout
        if self.dropout != 0:
            if training is not False and self.sparse_inputs:
                x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            elif training is not False:
                x = tf.nn.dropout(x, 1-self.dropout)

        Z = tf.stack([self.weights_[i] * x[i] for i in range(len(x))])

        output = tf.reduce_sum(Z, axis=0)

        if self.bias:
            output += self.bias

        return self.act(output)
