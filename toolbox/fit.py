# -*- coding: utf-8 -*-
""" fit.py

Created on 24-02-21

@author: Pierre-François De Plaen
"""

import time

from utils import * # toolbox.utils

import tensorflow as tf
from tensorflow.keras import optimizers

DETERMINISTIC = False

def preprocess_data(dataset, train_index, val_index, test_index):
    """
    Load the dataset and separate train, test and validation sets
    :param dataset:     (str) dataset path, datasets must be stored in data/datasets/DATASET_NAME/...
    :param train_index: (list of int) training indexes
    :param val_index:   (list of int) validation indexes
    :param test_index:  (list of int) test indexes

    :return:    adj:        (numpy array) adjacency: matrix NxN
                features:   (numpy array) node features: Nx(num features)
                y:          (numpy array) node classes: Nx(num classes)
                y_train, y_val, y_test: (numpy arrays) node classes for training, testing and validation
                train_mask, val_mask, test_mask: (numpy arrays) boolean masks for training, testing and validation
    """

    adj, features, y = load_dataset(dataset)

    # Create mask and y
    train_mask = sample_mask(train_index, y.shape[0])
    val_mask = sample_mask(val_index, y.shape[0])
    test_mask = sample_mask(test_index, y.shape[0])

    y_train = np.zeros(y.shape)
    y_val = np.zeros(y.shape)
    y_test = np.zeros(y.shape)
    y_train[train_mask, :] = y[train_mask, :]
    y_val[val_mask, :] = y[val_mask, :]
    y_test[test_mask, :] = y[test_mask, :]

    return adj, features, y, y_train, y_val, y_test, train_mask, val_mask, test_mask

def fit(dataset_name, train_index, val_index, test_index, model_params, model_utils, model_func, params_value, verbose=0, is_model_return=False):
    """
   Load the dataset and separate train, test and validation sets
   :param dataset:          (str) dataset path, datasets must be stored in data/datasets/DATASET_NAME/...
   :param train_index:      (list of int) training indexes
   :param val_index:        (list of int) validation indexes
   :param test_index:       (list of int) test indexes
   :param model_params:     (file import) parameters file of the model
   :param model_utils:      (file import) utilitaries file of the model
   :param model_func:       (function import) model definition
   :param params_value:     (dict) dictionary with some of the parameters of the model (the other are initialized with default value)
   :param verbose:          (int) verbosity for the execution
   :param is_model_return:  (bool) set to True to return the model

   :return:    result:  (dict) contains the train, test and validation accuracy as well as the number of epoch
                               (also returns the model when is_model_return=True)
   """
    # set random seed
    if DETERMINISTIC:
        seed = 123
        np.random.seed(seed)
        tf.random.set_seed(seed)

    # Fixed model parameters
    parameters = model_params.Params(**params_value)
    parameters.dataset = dataset_name

    # Define the TF function for the training step
    if parameters.num_epochs > 1:
        @tf.function
        def train_step(optimizer, model, features, adj, TF_train_label_batch, TF_train_mask_batch):
            with tf.GradientTape() as tape:
                loss, acc = model((features, TF_train_label_batch, TF_train_mask_batch, adj))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            #tf.print([el.shape for el in model.trainable_variables])

            return loss, acc
    else: # eager mode
        def train_step(optimizer, model, features, adj, TF_train_label_batch, TF_train_mask_batch):
            with tf.GradientTape() as tape:
                loss, acc = model((features, TF_train_label_batch, TF_train_mask_batch, adj))

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            return loss, acc

    # Load dataset and preprocess data
    adj, features, y, y_train, y_val, y_test, train_mask, val_mask, test_mask = preprocess_data(dataset_name, train_index, val_index, test_index)
    input_dim = features.shape[1]
    output_dim = y_train.shape[1]
    node_num = features.shape[0]

    adj, features, feed_data, num_feats_nnz = model_utils.model_preprocessing(parameters, adj, features)

    # Create model
    optimizer = optimizers.Adam(learning_rate=parameters.learning_rate)
    model = model_func(parameters, input_dim=input_dim, output_dim=output_dim, node_num=node_num, num_features_nonzero=num_feats_nnz, data=feed_data)

    cost_val = []

    #ll = list()
    # Train with batch model
    ttot = time.time()
    for epoch in range(parameters.num_epochs):
        t = time.time()
        train_loss = 0.
        train_acc = 0.
        np.random.shuffle(train_index)

        if parameters.batch_size:
            num_batch = len(train_index) // parameters.batch_size
        else:
            num_batch = 1

        for batch in range(num_batch):
            if verbose > 2:
                mini_duration = time.time()

            if parameters.batch_size:
                start = batch * parameters.batch_size
                end = min((batch + 1) * parameters.batch_size, len(train_index))
            else:
                start = 0
                end = len(train_index)

            if start < end:
                # Update y_train & train_mask
                y_train_batch = np.zeros((y.shape[0], y.shape[1]), float)
                y_train_batch[train_index[start:end], :] = y[train_index[start:end], :]
                train_mask_batch = np.zeros(y.shape[0], bool)
                train_mask_batch[train_index[start:end]] = True
                TF_train_label_batch = tf.convert_to_tensor(y_train_batch, dtype=tf.float32)
                TF_train_mask_batch = tf.convert_to_tensor(train_mask_batch)

                loss, acc = train_step(optimizer, model, features, adj, TF_train_label_batch, TF_train_mask_batch)
                train_loss += loss
                train_acc += acc

                if verbose > 2:
                    print("Minibatch {}/{}:".format(batch + 1, num_batch) + \
                          "Minibatch loss = {:.4f}, ".format(loss) + "Training Accuracy = {:.2f}".format(acc), "minibatch time=", "{:.5f}".format(time.time() - mini_duration))

        train_loss /= num_batch
        train_acc /= num_batch

        # Validation
        t_val = time.time()
        # Convert vectors to TF Vectors
        TF_val_label = tf.convert_to_tensor(y_val, dtype=tf.float32)
        TF_val_mask = tf.convert_to_tensor(val_mask)

        val_loss, val_acc = model((features, TF_val_label, TF_val_mask, adj), training=False)
        duration = (time.time() - t_val)

        #ll.append([train_loss, val_loss])

        cost_val.append(val_acc)

        # Print results
        if verbose > 1:
            print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(train_loss),
                  "train_acc=", "{:.5f}".format(train_acc), "val_loss=", "{:.5f}".format(val_loss),
                  "val_acc=", "{:.5f}".format(val_acc), "epoch time=", "{:.5f}".format(time.time() - t))

        if epoch > parameters.stop_window_size and cost_val[-1] <= np.mean(cost_val[-(parameters.stop_window_size + 1):-1]):
            if verbose > 1:
                print("Early stopping")
            break

    # Testing
    TF_test_label = tf.convert_to_tensor(y_test, dtype=tf.float32)
    TF_test_mask = tf.convert_to_tensor(test_mask)

    test_loss, test_acc = model((features, TF_test_label, TF_test_mask, adj), training=False)
    if verbose > 0:
        print("Test set results:", "cost=", "{:.5f}".format(test_loss),
              "accuracy=", "{:.5f}".format(test_acc), "TOTAL time=", "{:.5f}".format(time.time() - ttot))

    if epoch == parameters.num_epochs - 1 and verbose > 1:
        print("No early stopping")

    #print(repr(np.asarray(ll)))
    """for ws in model.trainable_variables:
        tf.print(ws.shape)
        tf.print(tf.reduce_sum(tf.abs(ws)))""" # TODO: remove !!!

    result = {
        'train_acc': train_acc,
        'valid_acc': val_acc,
        'test_acc': test_acc,
        'epoch': epoch
    }

    if is_model_return:
        result["model"] = model

    is_confusion_matrix = False
    if is_confusion_matrix: # confusion matrix
        test_preds = model.predict()
        test_preds = test_preds.numpy()
        pred_classes = np.argmax(test_preds[test_mask], axis=1)
        true_classes = np.argmax(y_test[test_mask], axis=1)
        confusion_matrix = tf.math.confusion_matrix(true_classes, pred_classes)
        result["confusion_matrix"] = confusion_matrix.numpy()

    return result
