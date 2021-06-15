# -*- coding: utf-8 -*-
""" util.py

Created on 17-02-21

@author: Pierre-François De Plaen
"""

import scipy.io
import os
import numpy as np
import sys

import errno

import pickle as pkl


def build_dict(keys, size=0, prefix=''):
    dic = {}
    for key in keys:
        if size == 0:
            dic[prefix + str(key)] = 0
        else:
            dic[prefix + str(key)] = np.zeros(size)
    return dic


def save_dict(receiver, sender, index=None):  # save the sender inside the receiver
    for key in sender.keys():
        if index == None:
            receiver[key] = sender[key]
        else:
            receiver[key][index] = sender[key]
    return receiver


def merge_dicts(dict_list, function, prefix=''):
    '''
    the function is called over all values that are at the same index
    (repeat for each key)
    '''
    keys = list(dict_list[0].keys())
    shape = dict_list[0][keys[0]].shape
    if shape == ():
        result = build_dict(keys, 0, prefix)
    else:
        result = build_dict(keys, shape, prefix)

    temp = None

    def loop(key, level, index):
        if level == len(shape):
            for k in range(len(dict_list)):
                if shape == ():
                    temp[k] = dict_list[k][key]
                else:
                    temp[k] = dict_list[k][key][tuple(index)]
            if shape == ():
                result[prefix + key] = function(temp)
            else:
                result[prefix + key][tuple(index)] = function(temp)
        else:
            for i_level in range(shape[level]):
                index[level] = i_level
                loop(key, level + 1, index)

    for key in keys:
        temp = np.zeros(len(dict_list))
        loop(key, 0, [0 for i in range(len(shape))])
    return result


def max_array_index(array):
    index = np.unravel_index(np.argmax(array, axis=None), array.shape)
    return index


def map_dict(dic, function, prefix=''):
    '''
    the function is called over all values
    '''
    result = {}
    for key in dic.keys():
        result[prefix + key] = function(dic[key])
    return result


def get_best_params(stats_params, params_range, ref_key='valid_acc'):
    best_params_index = max_array_index(stats_params[ref_key])
    best_params = get_params(best_params_index, params_range)
    best_stats_params = map_dict(stats_params, lambda x: x[tuple(best_params_index)])
    return best_stats_params, best_params_index, best_params


def get_params(params_index, params_range):
    c = -1

    def take_params(param_range):
        nonlocal c
        c += 1
        return param_range[params_index[c]]

    return map_dict(params_range, take_params)

# ----- FROM GCN/myutils.py -----

_path = os.path.dirname(os.path.abspath(__file__)) + '/'

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_dataset(dataset_name):
    dataset_mat = scipy.io.loadmat(_path + '../data/datasets/' + dataset_name + '/' + dataset_name + '.mat', squeeze_me=False)
    return dataset_mat['A'], dataset_mat['X'], dataset_mat['y_cs']


def unison_division_dataset(permutation_index, Y, n_fold):
    class_index = [[] for i in range(Y.shape[1])]
    nb_class = np.arange(Y.shape[1])
    y = Y * nb_class
    y = y.sum(axis=1)
    for i in range(len(permutation_index)):
        index = permutation_index[i]
        class_index[y[index]].append(index)
    f = 0
    fold_index = [[] for i in range(n_fold)]
    for i in range(len(class_index)):
        for j in range(len(class_index[i])):
            fold_index[f].append(class_index[i][j])
            f = (f + 1) % n_fold
    for i in range(n_fold):
        np.random.shuffle(fold_index[i])  # to mix class in each fold
    return fold_index


def load_cross_validation_run(dataset_name):
    run_directory = '../data/datasets/' + dataset_name + '/run/'
    run_pattern = '{}_index_run_{}.pkl'
    f = None
    runs = []
    stop = False
    r = 0

    while True:
        try:
            f = open(run_directory + run_pattern.format(dataset_name, r), "rb")
            run_stuct = pkl.load(f)
            runs.append(run_stuct)
        except IOError:
            stop = True
        finally:
            if f is not None:
                f.close()
            if stop:
                break
            else:
                r += 1
    return runs


def create_directory(directory):
    # create the directory if needed
    if not os.path.exists(directory):
        try:
            os.makedirs(directory)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


def A_to_P(A):
    d = A.sum(1)
    P = A / d[:, None]
    return P


def load_dataset_with_P(dataset_name):
    A, X, y_cs = load_dataset(dataset_name)
    P = A_to_P(A)
    return A, X, y_cs, P


def A_to_DAD(A):
    d = A.sum(1)
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return np.dot(np.dot(d_mat_inv_sqrt, A), d_mat_inv_sqrt)


def load_dataset_with_DAD(dataset_name):
    A, X, y_cs = load_dataset(dataset_name)
    DAD = A_to_DAD(A)
    return A, X, y_cs, DAD

# ----- FROM GCN/utils.py -----


def A_to_D(A):
    d = A.sum(1)
    D=np.diag(d)
    return D

def A_to_L(A):
    L=A_to_D(A)-A
    return L


def A_to_C(A):
    realmax = sys.maxsize
    C = 1/A
    C[np.isinf(C)] = realmax
    return C
