# -*- coding: utf-8 -*-
""" SVM models.py

Created on 24-02-21

@author: Pierre-François De Plaen
"""

import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.svm import SVC

class SVM(keras.Model):

    def __init__(self, parameters, input_dim, output_dim, node_num, num_features_nonzero, data=None, **kwargs):
        super(SVM, self).__init__(**kwargs)

        self.params = parameters
        self.input_dim = input_dim      # number of features
        self.output_dim = output_dim    # number of classes
        self.SVM_model = SVC(probability=True, kernel=parameters.svm_kernel)

        self.is_trained = False

    def call(self, inputs, training=None):
        X, Y, mask, _ = inputs
        X, Y, mask = (X.numpy(), Y.numpy(), mask.numpy())
        Y_class = np.argmax(Y, axis=1)

        if training != False:
            self.SVM_model.fit(X[mask, :], Y_class[mask])
            self.train_Y = Y
            self.train_mask = mask
            self.is_trained = True

            return tf.convert_to_tensor(0, dtype=tf.float32), tf.convert_to_tensor(1, dtype=tf.float32)
        elif self.is_trained:
            # Set the training values
            Y_old = np.zeros(Y.shape)
            Y_old[self.train_mask] = self.train_Y[self.train_mask]
            Y_class[self.train_mask] = np.argmax(self.train_Y[self.train_mask], axis=1)

            # Prediction
            Y_old[mask] = self.SVM_model.predict_proba(X[mask, :])

            self.outputs = Y_old

            # Cross entropy error
            loss = tf.convert_to_tensor(0, dtype=tf.float32)
            acc = np.mean(np.argmax(self.outputs[mask], axis=1) == Y_class[mask])
            acc = tf.convert_to_tensor(acc)

            return loss, acc
        else:
            RuntimeError("Model must be trained before making predictions")

    def predict(self):
        return self.outputs