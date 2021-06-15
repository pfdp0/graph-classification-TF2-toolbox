#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Sun Dec 29 16:04:12 2019

Code containing some usefull functions for data processing.

@author: Calbert Simon
"""

import os
import errno
import sys

import numpy as np
import scipy.io
import pickle as pkl
import scipy.sparse as sp
import networkx as nx

_path = os.path.dirname(os.path.abspath(__file__))+'/'


def load_dataset(dataset_name):
    dataset_mat = scipy.io.loadmat(_path+'dataset/'+dataset_name+'/'+dataset_name+'.mat', squeeze_me=False)
    return dataset_mat['A'], dataset_mat['X'], dataset_mat['y_cs']


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


def from_data_to_dataset(dataset_str, save_as):
    """
    Transforms data from data/untreated/ directory to matrices in data/datasets/

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("untreated/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    #test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    #test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    #features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    #labels[test_idx_reorder, :] = labels[test_idx_range, :]

    adj = adj.toarray()
    features = features.toarray()

    Y = np.argmax(labels, axis=1)+1

    scipy.io.savemat(
        'datasets/'+save_as+'/'+save_as+'.mat',
        {'A': adj,
         'X': features,
         'Y': Y, # just the class nb (starting at one), ex: 2 
         'y_cs': labels # the vector, ex: [0, 1, 0]
         },
        do_compression=True
    )
    print("Dataset matrices have been saved")
    # return adj, features, labels



if __name__ == '__main__':
    from_data_to_dataset("pubmed", "pubmedA500Xsym")
    
    
        