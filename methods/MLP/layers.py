# -*- coding: utf-8 -*-
""" MLP layers.py

Created on 04-04-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

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
        res = tf.matmul(x, y)
    return res


class Dense(layers.Layer):
    def __init__(self, input_dim, output_dim, num_features_nonzero,
                 dropout=0.,
                 is_sparse_inputs=False,
                 activation=tf.nn.relu,
                 bias=False,
                 featureless=False, **kwargs):
        super(Dense, self).__init__(**kwargs)
        self.dropout = dropout
        self.activation = activation
        self.is_sparse_inputs = is_sparse_inputs
        self.featureless = featureless
        self.bias = bias
        self.num_features_nonzero = num_features_nonzero

        self.weights_ = self.add_weight('weight', [input_dim, output_dim])

        if self.bias:
            self.bias = self.add_weight('bias', [output_dim])


    def call(self, inputs, training=True):
        x = inputs

        # dropout
        if self.dropout != 0:
            if training is not False and self.is_sparse_inputs:
                x = sparse_dropout(x, self.dropout, self.num_features_nonzero)
            elif training is not False:
                x = tf.nn.dropout(x, self.dropout)

        # transform
        output = dot(x, self.weights_, sparse=self.is_sparse_inputs)

        # bias
        if self.bias:
            output += self.vars['bias']

        return self.activation(output)