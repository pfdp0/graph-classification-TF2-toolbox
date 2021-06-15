# -*- coding: utf-8 -*-
""" AutoSVM models.py

Created on 24-02-21

@author: Pierre-François De Plaen
"""

import sys
import tensorflow as tf
from tensorflow import keras
import numpy as np
from sklearn.svm import SVC

class AutoSVM(keras.Model):
    """Autologistic SVM"""
    def __init__(self, parameters, input_dim, output_dim, node_num, num_features_nonzero, data=None, **kwargs):
        super(AutoSVM, self).__init__(**kwargs)

        self.params = parameters
        self.input_dim = input_dim      # number of features
        self.output_dim = output_dim    # number of classes
        self.SVM_model_1 = SVC(probability=True, kernel=parameters.svm_kernel)

        self.is_trained = False

    def call(self, inputs, training=None):
        X, Y, mask, P = inputs
        X, Y, mask, P = (X.numpy(), Y.numpy(), mask.numpy(), P.numpy())
        Y_class = np.argmax(Y, axis=1)

        if training != False:
            self.SVM_model_1.fit(X[mask, :], Y_class[mask])
            self.train_Y = Y
            self.train_mask = mask
            self.is_trained = True

            return tf.convert_to_tensor(0, dtype=tf.float32), tf.convert_to_tensor(1, dtype=tf.float32)
        elif self.is_trained:
            N = P.shape[0]

            # Set the training values
            Y_old = np.zeros(Y.shape)
            Y_old[self.train_mask] = self.train_Y[self.train_mask]
            Y_class[self.train_mask] = np.argmax(self.train_Y[self.train_mask], axis=1)

            # Initial predictions
            Y_old[mask] = self.SVM_model_1.predict_proba(X[mask, :])

            Y_new = np.zeros(Y.shape)
            iter = 0
            tol = 1e-8

            while np.sum(np.abs(Y_new - Y_old)) > tol :
                if iter == 0:
                    Y_new[self.train_mask] = Y_old[self.train_mask]
                else:
                    Y_old = Y_new

                # compute values of the autocovariates
                ac = np.dot(P, Y_old)

                # train AutoSVM
                X_auto = np.concatenate((ac, X), axis=1)
                SVM_model_2 = SVC(probability=True, kernel=self.params.svm_kernel)
                SVM_model_2.fit(X_auto[self.train_mask, :], Y_class[self.train_mask])

                # predict
                Y_new[mask] = SVM_model_2.predict_proba(X_auto[mask, :])

                iter += 1

            self.outputs = Y_new

            # Cross entropy error
            loss = tf.convert_to_tensor(0, dtype=tf.float32)
            acc = np.mean(np.argmax(self.outputs[mask], axis=1) == Y_class[mask])
            acc = tf.convert_to_tensor(acc)

            return loss, acc
        else:
            RuntimeError("Model must be trained before making predictions")

    def predict(self):
        return self.outputs