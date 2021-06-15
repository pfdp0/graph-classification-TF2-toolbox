# -*- coding: utf-8 -*-
""" GCN models.py

Created on 27-02-21
Inspired by: Github - dragen1860/GCN-TF2

@author: Pierre-François De Plaen
"""

import tensorflow as tf
from tensorflow import keras

from methods.GCN.layers import *

from toolbox.metrics import *

class GCN(keras.Model):

    def __init__(self, parameters, input_dim, output_dim, node_num, num_features_nonzero, data=None, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.params = parameters
        self.input_dim = input_dim
        self.output_dim = output_dim

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

            self.layers_.append(GraphConvolution(input_dim=layer_in,
                                            output_dim=layer_out, # OLD: output_dim=self.params.hidden1,
                                            num_features_nonzero=num_features_nonzero,
                                            activation=activation,
                                            dropout=self.params.dropout,
                                            is_sparse_inputs=sparse,
                                            autocast=False))

    def call(self, inputs, training=True):
        x, label, mask, support = inputs

        outputs = [x]

        for i,layer in enumerate(self.layers):
            hidden = layer((outputs[-1], support), training=training)
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