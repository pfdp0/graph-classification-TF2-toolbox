# -*- coding: utf-8 -*-
""" AutoSVM utils.py

Created on 24-02-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf
import numpy as np

def model_preprocessing(parameters, adj, features):
    adj = adj / np.sum(adj, axis=0) # Transition probas

    features = tf.convert_to_tensor(features, dtype=tf.float32)
    adj = tf.convert_to_tensor(adj, dtype=tf.float32)

    return adj, features, None, None
