# -*- coding: utf-8 -*-
""" GENERAL metrics.py

Created on 27-02-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf

from tensorflow.keras.losses import Loss

class CustomLosses(Loss):
    """ Extension of Keras.losses:
    custom Keras losses with a mask
    """
    def __init__(self, loss_name, loss_parameters=None, **kwargs):
        """
        :param loss_name:       (str) loss name, all losses below are allowed (name of the loss without the prefix "masked_"
        :param loss_parameters: (dict) with the parameters of the loss function
        """
        super(CustomLosses, self).__init__(**kwargs)

        full_loss_name = "masked_"+str(loss_name)
        try:
            self.loss_function = eval(full_loss_name)
            self.loss_parameters = loss_parameters
        except NameError:
            print(f"Loss function '{full_loss_name}' is not implemented.")
            raise

    def __call__(self, y_pred, y_true, sample_weight=None):
        """
        :param y_pred:          (tf.tensor of float) predicted labels (NxC)
        :param y_true:          (tf.tensor of float) true labels in one-hot encoding (NxC)
        :param sample_weight:   (tf.tensor of bool) mask for the regularization (N)
        """
        if sample_weight == None:
            sample_weight = tf.ones(y_true.shape[0], dtype=tf.bool)

        if self.loss_parameters is not None:
            return self.loss_function(y_pred, y_true, sample_weight, **self.loss_parameters)
        else:
            return self.loss_function(y_pred, y_true, sample_weight)


def masked_softmax_cross_entropy(preds, labels, mask):
    """Softmax cross-entropy loss with masking"""
    loss = tf.nn.softmax_cross_entropy_with_logits(logits=preds, labels=labels)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    loss *= mask
    return tf.reduce_mean(loss)


def masked_regu_L(preds, labels, mask, A, d, norm, lambda_coef):
    """Graph regularization with Laplacian matrix
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    :param A:       (tf.tensor of float) adjacency matrix of the graph (NxX)
    :param d:       (tf.tensor of float) degree matrix of the graph stored as (Nx1 matrix)
    :param norm:        (int) regularization norm: 1 or 2
    :param lambda_coef: (float) weight coefficient for the regularization
    """
    mask = tf.cast(mask, dtype=tf.float32)
    preds = tf.boolean_mask(preds, mask)
    A_mask = tf.boolean_mask(tf.boolean_mask(A, mask), mask, axis=1)
    d_mask = tf.boolean_mask(d, mask)

    diff = (d_mask * preds) - tf.matmul(A_mask, preds)

    if norm == 1:
        sq = tf.math.abs(diff) # norm 1
    elif norm == 2:
        sq = tf.math.square(diff) # norm 2
    else:
        Exception(f"Norm {norm} is not valid, please select norm 1 or 2")

    norm_logits = tf.math.reduce_sum(sq, axis=1)

    loss = tf.reduce_mean(lambda_coef*norm_logits)

    return loss


def masked_regu_P(preds, labels, mask, P, norm, lambda_coef):
    """Graph regularization with Probability transition matrix
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    :param P:       (tf.tensor of float) probability transition matrix of the graph (NxX)
    :param norm:        (int) regularization norm: 1 or 2
    :param lambda_coef: (float) weight coefficient for the regularization
    """
    mask = tf.cast(mask, dtype=tf.float32)
    P_mask = tf.boolean_mask(tf.boolean_mask(P, mask), mask, axis=1)
    preds = tf.boolean_mask(preds, mask)

    diff = preds - tf.matmul(P_mask, preds)
    if norm == 1:
        sq = tf.math.abs(diff) # norm 1
    elif norm == 2:
        sq = tf.math.square(diff) # norm 2
    else:
        Exception(f"Norm {norm} is not valid, please select norm 1 or 2")
    norm_logits = tf.math.reduce_sum(sq,axis=1)

    loss = tf.reduce_mean(lambda_coef*norm_logits)

    return loss


def masked_accuracy(preds, labels, mask):
    """Masked accuracy
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    correct_prediction = tf.equal(tf.argmax(preds, 1), tf.argmax(labels, 1))
    accuracy_all = tf.cast(correct_prediction, tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)
    mask /= tf.reduce_mean(mask)
    accuracy_all *= mask
    return tf.reduce_mean(accuracy_all)


mae = tf.keras.losses.MeanAbsoluteError()
def masked_L1_loss(preds, labels, mask):
    """Masked L1 loss
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    return mae(labels, preds, sample_weight=mask)

def masked_softmax_L1_loss(preds, labels, mask):
    """Masked L1 loss with softmax activation on the predictions
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    return mae(labels, tf.nn.softmax(preds), sample_weight=mask)


mse = tf.keras.losses.MeanSquaredError()
def masked_L2_loss(preds, labels, mask):
    """Masked L2 loss
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    return mse(labels, preds, sample_weight=mask)

def masked_softmax_L2_loss(preds, labels, mask):
    """Masked L2 loss with softmax activation on the predictions
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    return mse(labels, tf.nn.softmax(preds), sample_weight=mask)


def masked_hinge_loss(preds, labels, mask):
    """Masked hinge loss
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    #y_pred = tf.cast(preds, dtype=tf.float32)
    y_pred = 2*tf.cast(preds, dtype=tf.float32) - 1
    y_true = 2*tf.cast(labels, dtype=tf.float32) - 1 # converts to {-1,1} labels
    mask = tf.cast(mask, dtype=tf.float32)

    loss = tf.reduce_mean(tf.nn.relu(1 - y_true * y_pred), axis=1)
    loss *= mask

    return tf.reduce_mean(loss, axis=0)

def masked_squared_hinge_loss(preds, labels, mask):
    """Masked hinge loss squared
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    # y_pred = tf.cast(preds, dtype=tf.float32)
    y_pred = 2 * tf.cast(preds, dtype=tf.float32) - 1
    y_true = 2*tf.cast(labels, dtype=tf.float32) - 1 # convert to {-1,1} labels
    mask = tf.cast(mask, dtype=tf.float32)

    loss = tf.reduce_mean(tf.pow(tf.nn.relu(1 - y_true * y_pred), 2), axis=1)
    loss *= mask

    return tf.reduce_mean(loss, axis=0)

def masked_cubed_hinge_loss(preds, labels, mask):
    """Masked hinge loss cubed
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    """
    # y_pred = tf.cast(preds, dtype=tf.float32)
    y_pred = 2 * tf.cast(preds, dtype=tf.float32) - 1
    y_true = 2*tf.cast(labels, dtype=tf.float32) - 1 # convert to {-1,1} labels
    mask = tf.cast(mask, dtype=tf.float32)

    loss = tf.reduce_mean(tf.pow(tf.nn.relu(1 - y_true * y_pred), 3), axis=1)
    loss *= mask

    return tf.reduce_mean(loss, axis=0)


def masked_softmax_focal_loss(preds, labels, mask, gamma=0.):
    """Masked focal loss with softmax activation on the predictions
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    :param gamma:   (float) focusing parameter >= 0; is equivalent to cross-entropy loss when gamma=0
    """
    preds = tf.nn.softmax(preds)
    labels = tf.cast(labels, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    loss = - tf.reduce_sum(labels * (1-preds)**gamma * tf.math.log(preds), axis=1)
    loss *= mask

    return tf.reduce_mean(loss)


def masked_softmax_CB_cross_entropy(preds, labels, mask, beta=0., N=()):
    """Masked Class Balanced cross entropy loss with softmax activation on the predictions
    :param preds:   (tf.tensor of float) predicted labels (NxC)
    :param labels:  (tf.tensor of float) true labels in one-hot encoding (NxC)
    :param mask:    (tf.tensor of bool) mask for the regularization (N)
    :param beta:   (float) focusing parameter >= 0; is equivalent to cross-entropy loss when gamma=0
    """
    preds = tf.nn.softmax(preds)
    labels = tf.cast(labels, dtype=tf.float32)
    mask = tf.cast(mask, dtype=tf.float32)

    loss = - tf.reduce_sum(labels * tf.math.log(preds), axis=1) # TODO: EDIT (not done yet...)
    loss *= mask

    return tf.reduce_mean(loss)