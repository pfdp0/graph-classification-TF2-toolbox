# -*- coding: utf-8 -*-
""" GWNN layers.py

Created on 05-03-21

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


class Wavelet_Convolution(layers.Layer):
    """Wavelet convolution layer."""
    def __init__(self, node_num,weight_normalize,input_dim, output_dim, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False,
                 featureless=False, num_features_nonzero=None, sparse_mode=True, **kwargs):
        super(Wavelet_Convolution, self).__init__(**kwargs)

        self.node_num = node_num
        self.weight_normalize = weight_normalize
        self.act = act
        self.sparse_inputs = sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero
        self.dropout = dropout
        self.sparse_mode = sparse_mode

        self.weights_ = []
        w = self.add_weight('weights_' + str(0), [input_dim, output_dim])
        self.weights_.append(w)

        # diag filter kernel (stored in a Nx1 tensor for efficient training and memory)
        self.kernel_ = self.add_weight('kernel', [self.node_num,1], initializer=tf.ones_initializer())

        if self.bias:
            self.bias = self.add_weight('bias', [output_dim])

    def call(self, inputs, training=True):
        x, phis_ = inputs

        # dropout
        if self.dropout != 0:
            if training is not False and self.sparse_inputs:
                x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)
            elif training is not False:
                x = tf.nn.dropout(x, 1-self.dropout)

        # convolve # TODO: solve phis_ ?!!!
        X_prime = dot(x, self.weights_[0], sparse=self.sparse_inputs)
        X_prime_transform = dot(phis_[0], X_prime, sparse=self.sparse_mode)
        output = dot(phis_[1], self.kernel_ * X_prime_transform, sparse=self.sparse_mode)

        if self.bias:
            output += self.bias

        return self.act(output)
