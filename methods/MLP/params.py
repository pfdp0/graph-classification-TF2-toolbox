# -*- coding: utf-8 -*-
""" MLP params.py

Created on 04-04-21

@author: Pierre-François De Plaen
"""

from tensorflow import nn

import toolbox.metrics as metrics

class Params(object):
    def __init__(self, hidden_layer_sizes=None, activation=nn.relu,
                 num_epochs=200, batch_size=64, learning_rate=0.01,
                 loss_function="softmax_cross_entropy", loss_parameters=None,
                 dropout=0, stop_window_size=10, loss_regularization=None, lambda_coef=0, regu_norm=1, weight_decay=0,
                 preprocess_feats=False):
        """Params of the MLP model"""

        if hidden_layer_sizes == None:
            hidden_layer_sizes = []
        elif not isinstance(hidden_layer_sizes, list):
            hidden_layer_sizes = [hidden_layer_sizes]

        self.num_epochs = num_epochs
        self.stop_window_size = stop_window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.dropout = dropout
        self.preprocess_feats = preprocess_feats

        self.activation = activation
        self.num_hidden_layers = len(hidden_layer_sizes)
        self.hidden_layer_sizes = hidden_layer_sizes

        self.loss_regularization = loss_regularization  # allowed: P, L (Laplacian) or None
        self.lambda_coef = lambda_coef
        self.regu_norm = regu_norm                      # norm for loss regularization

        self.weight_decay = weight_decay  # was 5e-4

        self.loss_function = metrics.CustomLosses(loss_function, loss_parameters=loss_parameters)
