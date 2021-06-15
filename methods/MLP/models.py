# -*- coding: utf-8 -*-
""" MLP models.py

Created on 04-04-21

@author: Pierre-François De Plaen
"""

import sys
from tensorflow import keras
import numpy as np

from methods.MLP.layers import *
from toolbox.metrics import *

class MLP(keras.Model):

    def __init__(self, parameters, input_dim, output_dim, node_num, num_features_nonzero, data=None, **kwargs):
        super(MLP, self).__init__(**kwargs)

        self.params = parameters
        self.input_dim = input_dim      # number of features
        self.output_dim = output_dim    # number of classes
        self.num_features_nonzero = num_features_nonzero

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

        if parameters.num_hidden_layers == 0:
            layers_in_out_dims = [[input_dim, output_dim]]
        else:
            layers_in_out_dims = list()
            for i in range(parameters.num_hidden_layers):
                layers_in_out_dims.append([parameters.hidden_layer_sizes[i-1], parameters.hidden_layer_sizes[i]])

            # edit first layer and add last layer
            layers_in_out_dims[0][0] = input_dim
            layers_in_out_dims.append([parameters.hidden_layer_sizes[-1], output_dim])

        self.layers_ = []
        for i, (layer_in, layer_out) in enumerate(layers_in_out_dims):
            sparse = True if i == 0 else False
            activation = self.params.activation if i != parameters.num_hidden_layers else lambda x: x

            self.layers_.append(Dense(input_dim=layer_in,
                                     output_dim=layer_out,
                                     num_features_nonzero=num_features_nonzero,
                                     activation=activation,
                                     dropout=self.params.dropout,
                                     is_sparse_inputs=sparse))


    def call(self, inputs, training=True):
        x, label, mask, _ = inputs

        hidden = x

        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, training=training)

        self.outputs = hidden

        # Weight decay loss
        loss = tf.zeros([])
        if self.params.weight_decay != 0:
            for var in self.layers_[0].trainable_variables:
                loss += self.params.weight_decay * tf.nn.l2_loss(var)

        # Base loss
        loss += self.loss_function(self.outputs, label, mask)

        # Graph regularization loss
        if self.params.loss_regularization == "P":
            loss += masked_regu_P(self.outputs, label, mask,
                                  self.P, self.norm, self.lambda_coef)
        elif self.params.loss_regularization == "L":
            loss += masked_regu_L(self.outputs, label, mask,
                                  self.A, self.d, self.norm, self.lambda_coef)

        acc = masked_accuracy(self.outputs, label, mask)

        return loss, acc

    def predict(self):
        return tf.nn.softmax(self.outputs)