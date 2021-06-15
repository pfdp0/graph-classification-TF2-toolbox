# -*- coding: utf-8 -*-
""" M_GWNN layers.py

Created on 20-05-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf

import toolbox.metrics as metrics

class Params(object):
    def __init__(self, hidden_layer_sizes=None, activation=tf.nn.relu,
                 laplacian_normalize=False, weight_normalize=False, sparse_threshold=1e-5, sparse_ness=True,
                 num_epochs=200, batch_size=64, learning_rate=0.01, sparse_mode=False,
                 loss_function="softmax_cross_entropy", loss_parameters=None,
                 dropout=0, stop_window_size=10, loss_regularization=None, weight_decay=0, lambda_coef=0, regu_norm=1,
                 preprocess_feats=False):
        """Params of the M_GWNN model"""
        self.sparse_mode = sparse_mode

        self.num_epochs = num_epochs
        self.stop_window_size = stop_window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        if hidden_layer_sizes == None:
            hidden_layer_sizes = []
        elif not isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = [hidden_layer_sizes]

        self.num_hidden_layers = len(hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes

        self.dropout = dropout
        self.activation = activation
        self.preprocess_feats = preprocess_feats

        self.weight_decay = weight_decay

        self.loss_regularization = loss_regularization  # allowed: P, L (Laplacian) or None
        self.lambda_coef = lambda_coef
        self.regu_norm = regu_norm  # norm for loss regularization

        self.loss_function = metrics.CustomLosses(loss_function, loss_parameters=loss_parameters)

        # GWNN internal parameters
        self.wavelet_s = [0, 4**(-4), 4**(-2), 4**(0)]
        self.num_wavelet_s = len(self.wavelet_s)

        self.sparse_threshold = sparse_threshold
        self.laplacian_normalize = laplacian_normalize
        self.weight_normalize = weight_normalize
        self.sparse_ness = sparse_ness
