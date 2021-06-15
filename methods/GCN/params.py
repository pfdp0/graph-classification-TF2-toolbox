# -*- coding: utf-8 -*-
""" GCN params.py

Created on 23-02-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf

import toolbox.metrics as metrics

class Params(object):
    def __init__(self, num_supports=1, hidden_layer_sizes=None, activation=tf.nn.relu,
                 num_epochs=200, batch_size=64, learning_rate=0.01,
                 loss_function="softmax_cross_entropy", loss_parameters=None,
                 dropout=0, stop_window_size=10, loss_regularization=None, lambda_coef=0, regu_norm=1, weight_decay=0,
                 preprocess_feats=False):
        """ Params of the GCN model """
        if hidden_layer_sizes == None:
            hidden_layer_sizes = []
        elif not isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = [hidden_layer_sizes]

        self.num_hidden_layers = len(hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes # OLD: self.hidden1 = hidden

        self.num_epochs = num_epochs
        self.stop_window_size = stop_window_size
        self.batch_size = batch_size
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.num_supports = num_supports
        self.learning_rate = learning_rate
        self.model = "gcn"
        self.activation = activation
        self.preprocess_feats = preprocess_feats

        self.loss_regularization = loss_regularization # allowed: P, L (Laplacian) or None
        self.lambda_coef = lambda_coef
        self.regu_norm = regu_norm                      # norm for loss regularization

        self.loss_function = metrics.CustomLosses(loss_function, loss_parameters=loss_parameters)

