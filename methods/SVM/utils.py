# -*- coding: utf-8 -*-
""" SVM utils.py

Created on 24-02-21

@author: Pierre-François De Plaen
"""

import tensorflow as tf

def model_preprocessing(parameters, adj, features):
    features = tf.convert_to_tensor(features, dtype=tf.float32)

    return None, features, None, None
