# -*- coding: utf-8 -*-
""" SVM params.py

Created on 24-02-21

@author: Pierre-François De Plaen
"""

class Params(object):
    def __init__(self, stop_window_size=1, num_epochs=1, batch_size=False,
    learning_rate=0.05, svm_kernel='rbf'):
        """Params of the SVM model"""
        self.num_epochs = num_epochs
        self.stop_window_size = stop_window_size
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.svm_kernel = svm_kernel
