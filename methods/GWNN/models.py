# -*- coding: utf-8 -*-
""" GWNN models.py

Created on 05-03-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf
from tensorflow import keras

from methods.GWNN.layers import *
from toolbox.metrics import *

class GWNN(keras.Model):
    def __init__(self, parameters, input_dim, output_dim, node_num, num_features_nonzero, data=None, **kwargs):
        super(GWNN, self).__init__(**kwargs)

        self.weight_normalize = parameters.weight_normalize
        self.input_dim = input_dim
        self.node_num = node_num
        self.output_dim = output_dim
        self.params = parameters
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

            self.layers_.append(Wavelet_Convolution(node_num=self.node_num,
                                           weight_normalize=self.weight_normalize,
                                           input_dim=layer_in,
                                           output_dim=layer_out,
                                           act=activation,
                                           dropout=parameters.dropout,
                                           sparse_inputs=sparse,
                                           sparse_mode=parameters.sparse_mode,
                                           num_features_nonzero=num_features_nonzero))

    def call(self, inputs, training=True):
        x, label, mask, adj = inputs

        outputs = [x]

        for layer in self.layers_:
            hidden = layer((outputs[-1], adj), training=training)
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