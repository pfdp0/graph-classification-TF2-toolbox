# -*- coding: utf-8 -*-
""" DCNN models.py

Created on 17-02-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf
from tensorflow import keras

from methods.DCNN.layers import *
from toolbox.metrics import *




class DCNN(keras.Model):
    def __init__(self, parameters, input_dim, output_dim, node_num, num_features_nonzero, data=None, **kwargs):
        super(DCNN, self).__init__(**kwargs)

        self.params = parameters
        self.input_dim = input_dim # num features
        self.output_dim = output_dim # num classes
        self.feature_num = num_features_nonzero

        self.loss_function = parameters.loss_function

        if self.params.loss_regularization == "P":
            self.P = data["P"]
            self.norm = self.params.regu_norm
            self.lambda_coef = self.params.lambda_coef
        elif self.params.loss_regularization == "L":
            self.A = data["A"]
            self.d = data["d"]
            self.norm = self.params.regu_norm
            self.lambda_coef = self.params.lambda_coef

        self.layers_ = []

        self.layers_.append(DiffusionConvolution(input_dim=self.input_dim,
                                             num_features_nonzero=num_features_nonzero,
                                             hops_num=self.params.hops_num,
                                             activation=self.params.activation,
                                             dropout=self.params.dropout,
                                             sparse_mode=self.params.sparse_mode))

        self.layers_.append(DiffusionDense(input_dim=self.input_dim, output_dim=self.output_dim,
                                           hops_num=self.params.hops_num,
                                           activation=lambda x: x))

    def call(self, inputs, training=True):
        x, label, mask, Pt = inputs

        outputs = [x]

        for i,layer in enumerate(self.layers):
            if i == 0:
                hidden = layer((outputs[-1], Pt), training=training)
            else:
                hidden = layer(outputs[-1], training=training)
            outputs.append(hidden)

        self.outputs = outputs[-1]

        # Weight decay loss
        loss = tf.zeros([])
        if self.params.weight_decay != 0:
            for var in self.layers_[0].trainable_variables:
                loss += self.params.weight_decay * tf.nn.l2_loss(var)

        # Base loss
        loss += self.loss_function(outputs[-1], label, mask)

        # Graph regularization loss
        if self.params.loss_regularization == "P":
            loss += masked_regu_P(outputs[-1], label, mask,
                                  self.P, self.norm, self.lambda_coef)
        elif self.params.loss_regularization == "L":
            loss += masked_regu_L(outputs[-1], label, mask,
                                  self.A, self.d, self.norm, self.lambda_coef)

        acc = masked_accuracy(outputs[-1], label, mask)

        return loss, acc

    def predict(self):
        return tf.nn.softmax(self.outputs)