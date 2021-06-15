# -*- coding: utf-8 -*-
""" DCNN layers.py

Created on 18-02-21

@author: Pierre-François De Plaen
"""

import time
from utils import *

import tensorflow as tf
from tensorflow.keras import layers


class DiffusionConvolution(layers.Layer):
    "Diffusion convolution layer"

    def __init__(self, input_dim, num_features_nonzero, hops_num,
                 dropout=0., activation=tf.nn.relu, sparse_mode=True, **kwargs):
        super(DiffusionConvolution, self).__init__(autocast=False, **kwargs)

        self.activation = activation
        self.dropout = dropout
        self.num_features_nonzero = num_features_nonzero
        self.hops_num = hops_num
        self.input_dim = input_dim
        self.sparse_mode = sparse_mode

        self.W = self.add_weight(name='W_c', shape=[hops_num + 1, input_dim])

    def call(self, inputs, training=True):
        X, Pt = inputs

        # dropout
        if self.dropout != 0:
            if training is not False:
                X = tf.nn.dropout(X, self.dropout)


        interm = list()
        for i in range(self.hops_num + 1):
            if self.sparse_mode:
                interm.append(self.W[i] * tf.sparse.sparse_dense_matmul(Pt[i], X))
            else:
                interm.append(self.W[i] * tf.matmul(Pt[i], X))

        return self.activation(tf.stack(interm))


class DiffusionDense(layers.Layer):
    "Diffusion dense layer"

    def __init__(self, input_dim, output_dim, hops_num,
                 activation=lambda x: x, **kwargs):
        super(DiffusionDense, self).__init__(autocast=False, **kwargs)

        self.activation = activation

        self.W = self.add_weight('W_d_' + str(0), [hops_num+1, input_dim, output_dim])

    def call(self, inputs, training=True):
        X = inputs

        Z = tf.reduce_sum(tf.matmul(X, self.W), axis=0)

        return self.activation(Z)