# -*- coding: utf-8 -*-
""" experiments.py

Created on 16-03-21

@author: Pierre-François De Plaen
"""
import sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # run on CPU only

sys.path.append("C:/Users/pfdp/Documents/UCL/Thèse/CNNs-Graphs-toolbox")

from main import *

"""
from methods.GCN.models import GCN
import methods.GCN.params as GCNparams
import methods.GCN.utils as GCNutils

from methods.DCNN.models import DCNN
import methods.DCNN.params as DCNNparams
import methods.DCNN.utils as DCNNutils

from methods.AutoSVM.models import AutoSVM
import methods.AutoSVM.params as AutoSVMparams
import methods.AutoSVM.utils as AutoSVMutils

from methods.SVM.models import SVM
import methods.SVM.params as SVMparams
import methods.SVM.utils as SVMutils
"""
from methods.DCNN.models import DCNN_equiv
import methods.DCNN.params as DCNNparams
import methods.DCNN.utils as DCNNutils

from methods.MLP.models import MLP
import methods.MLP.params as MLPparams
import methods.MLP.utils as MLPutils

from methods.GWNN.models import GWNN
import methods.GWNN.params as GWNNparams
import methods.GWNN.utils as GWNNutils

from methods.M_GWNN.models import M_GWNN
import methods.M_GWNN.params as M_GWNNparams
import methods.M_GWNN.utils as M_GWNNutils

from methods.M_GWNN_band.models import M_GWNN_band
import methods.M_GWNN_band.params as M_GWNN_bandparams
import methods.M_GWNN_band.utils as M_GWNN_bandutils

# datasets_list_A_end = ['myciteseerA100Xsym', 'mycoraA100Xsym', 'myfacebookA100X107', 'myfacebookA100X1684', 'myfacebookA100X1912', 'mywikipediaA100Xsym']
datasets_list = ['mytexasA100Xsym', 'mywashingtonA100Xsym', 'mywisconsinA100Xsym', 'mycornellA100Xsym', 'myciteseerA100Xsym', 'mycoraA100Xsym', 'myfacebookA100X107', 'myfacebookA100X1684', 'myfacebookA100X1912', 'mywikipediaA100Xsym', 'myamazonphotoA100Xsym', 'myamazoncomputersA100Xsym']
# datasets_list_B = ['myamazonphotoA100Xsym', 'myamazoncomputersA100Xsym']
datasets_list_Amazon1 = ['myamazonphotoA100Xsym']
datasets_list_Amazon2 = ['myamazoncomputersA100Xsym']

datasets_list_Amazon = datasets_list_Amazon1 + datasets_list_Amazon2


DCNN_fine_tuning = {
    'DCNN': {
        'function': DCNN, # DCNN_equiv: is equivalent, but quicker ;)
        'params_fct': DCNNparams,
        'utils_fct': DCNNutils,
        'params_range': {
            'loss_function' : ["softmax_cross_entropy", "cubed_hinge_loss"],
            'loss_regularization' : ['P'],
            'sparse_mode': [False]
        }
    }
}


""""'DCNN_equiv': {
        'function': DCNN_equiv,
        'params_fct': DCNNparams,
        'utils_fct': DCNNutils,
        'params_range': {'activation': [tf.nn.tanh],
                         'learning_rate': [0.001, 0.01],
                         'hops': [1, 2, 3],
                         'sparse_mode': [True]}
    }"""

"""'DCNN': {
        'function': DCNN,
        'params_fct': DCNNparams,
        'utils_fct': DCNNutils,
        'params_range': {'activation': [tf.nn.tanh],
                         'learning_rate': [0.001, 0.01],
                         'hops': [1, 2, 3],
                         'sparse_mode': [False]}
    },"""


run_on_best_params(DCNN_fine_tuning, datasets_list[:4], folder_path="/results/parameters_fitting/", location="FINAL_TUNING_BIS")
