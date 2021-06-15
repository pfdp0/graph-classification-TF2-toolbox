# -*- coding: utf-8 -*-
""" DCNN models.py

Created on 23-02-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf

import toolbox.metrics as metrics

class Params(object):
    def __init__(self, hops=3, activation=tf.nn.relu,
                 learning_rate=0.01, num_epochs=200, batch_size=64, sparse_mode=False,
                 loss_function="softmax_cross_entropy", loss_parameters=None,
                 dropout=0, stop_window_size=10, loss_regularization=None, lambda_coef=0, regu_norm=1, weight_decay=0,
                 preprocess_feats=False):
        ''' Params of the DCNN model '''
        self.sparse_mode = sparse_mode

        self.hops_num = hops
        self.stop_window_size = stop_window_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.activation = activation
        self.preprocess_feats = preprocess_feats

        self.loss_regularization = loss_regularization  # allowed: P, L (Laplacian) or None
        self.lambda_coef = lambda_coef
        self.regu_norm = regu_norm                      # norm for loss regularization

        self.weight_decay = weight_decay

        self.loss_function = metrics.CustomLosses(loss_function, loss_parameters=loss_parameters)

