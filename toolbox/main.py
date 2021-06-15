# -*- coding: utf-8 -*-
""" main.py

Created on 17-02-21

@author: Pierre-François De Plaen
"""

# Import Python modules
import os
import re
from random import randint

# Remove TensorFlow information messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_CPP_MIN_VLOG_LEVEL'] = '3'

# Import TensorFlow and Pandas
import tensorflow as tf
import pandas as pd

# Import the toolbox
from utils import *
from fit import fit

# Graph Convolutional Network (GCN)
from methods.GCN.models import GCN
import methods.GCN.params as GCNparams
import methods.GCN.utils as GCNutils

# Diffusion Convolutional Neural Network (DCNN)
from methods.DCNN.models import DCNN
import methods.DCNN.params as DCNNparams
import methods.DCNN.utils as DCNNutils

# Graph Wavelet Neural Network (GWNN)
from methods.GWNN.models import GWNN
import methods.GWNN.params as GWNNparams
import methods.GWNN.utils as GWNNutils


def validate_model(model_name, model_info, dataset_name, verbose=1, location=None):
    """
    Run 10 times with 5 folds the model passed in argument on the datset passed in argument.
    All combinations of the params_range are run, the best results are saved in the folder *location*
    :param model_name:      (str) name of the model
    :param model_info:      (dict) dictionary with the information about the model (params_range, params_fct, utils_fct and function)
    :param dataset_name:    (str) dataset path, datasets must be stored in data/datasets/DATASET_NAME/...
    :param verbose:         (int) verbosity for the execution
    :param location:        (str) string with the folder location for saving the results. Creates a new random folder if None
    :return:                /
    """

    ### Create the directory for saving the results if it doesn't exist yet ###
    if location is None:
        location = str(randint(1000, 9999))
    save_directory = os.path.dirname(os.path.abspath(__file__)) + '/results/' + location + '/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    ### Initialize the parameters ###
    params_range = model_info['params_range']
    model_params = model_info['params_fct']
    model_utils = model_info['utils_fct']
    model_func = model_info['function']

    params_key = list(params_range.keys())
    n_params = len(params_key)
    keys_stats = ['train_acc', 'valid_acc', 'test_acc', 'epoch']
    grid_dim = tuple([len(val) for val in params_range.values()])

    ### Save results in a Pandas DataFrame ###
    column_names = ['run_id'] + keys_stats + params_key
    runs_results = pd.DataFrame(columns=column_names)

    ### Cross-validation: 10 runs of 5 folds ###
    runs = load_cross_validation_run(dataset_name)  # load all runs index associated to the given dataset
    best_params_occ = np.zeros(grid_dim)
    n_runs = len(runs)

    global_stats_params = [None] * n_runs
    for i in range(n_runs):
        if verbose > 0:
            print("RUN : " + str(i))
        run = runs[i]
        t_v_fold_index = run['sub_fold']
        test_index = run['test']
        n_fold = len(t_v_fold_index)
        stats_params = [None] * n_fold
        for j in range(n_fold):
            if verbose > 0:
                print("FOLD : " + str(j))
            folds = [k for k in range(n_fold)]
            folds.remove(j)
            train_index = sum([t_v_fold_index[z] for z in folds], [])
            val_index = t_v_fold_index[j]

            stat_params = build_dict(keys_stats, grid_dim)

            def loop(dataset_name, train_index, val_index, test_index, grid_dim, stats_params, params_range, params_key,
                     level, params_value, index):
                if level == n_params:
                    # fit the model
                    if verbose > 0:
                        print(f"Parameters: {params_value}")
                    result = fit(dataset_name, train_index, val_index, test_index, model_params, model_utils, model_func, params_value=params_value, verbose=verbose)
                    stats_params = save_dict(stat_params, result, tuple(index))
                else:
                    for i_level in range(grid_dim[level]):
                        key = params_key[level]
                        params_value[key] = params_range[key][i_level]
                        index[level] = i_level
                        loop(dataset_name, train_index, val_index, test_index, grid_dim, stats_params, params_range,
                             params_key, level + 1, params_value, index)

            params_value = build_dict(params_key)
            loop(dataset_name, train_index, val_index, test_index, grid_dim, stats_params, params_range, params_key, 0,
                 params_value, [0 for i in range(n_params)])
            stats_params[j] = stat_params
        # mean performance for each params on all folds
        mean_stats_params = merge_dicts(stats_params, np.mean)
        global_stats_params[i], best_params_index, best_params = get_best_params(mean_stats_params, params_range,
                                                                                 ref_key='valid_acc')
        best_params_occ[best_params_index] += 1
        runs_results.loc[len(runs_results)] = [str(i)] + list(global_stats_params[i].values()) + list(best_params.values())
    # mean and std of performance over all runs
    mean_global_stats_params = merge_dicts(global_stats_params, np.mean)
    std_global_stats_params = merge_dicts(global_stats_params, np.std)
    # best params (most of occurrences)
    best_params_occ /= n_runs
    best_params_index = max_array_index(best_params_occ)
    best_params_mode = best_params_occ[tuple(best_params_index)]
    best_params = get_params(best_params_index, params_range)

    ### Save DataFrame to file ###
    runs_results.loc[len(runs_results)] = ['MEAN'] + list(mean_global_stats_params.values()) + [None for _ in range(len(best_params))]
    runs_results.loc[len(runs_results)] = ['STD'] + list(std_global_stats_params.values()) + [None for _ in range(len(best_params))]
    if verbose > 0:
        print(runs_results)

    csv_name = "{}_{}.csv".format(dataset_name, model_name)
    aw = 'a' if os.path.exists(save_directory+csv_name) else 'w'

    with open(save_directory+csv_name, aw) as f:
        f.write("-- Parameters range: \n")
        for p_name, p_vals in params_range.items():
            f.write(f"{p_name} ; {[v for v in p_vals]}\n")
        f.write("\n")

        f.write("-- Best parameters: \n") # best params (most of occurences)
        best_params_occ /= n_runs
        best_params_index = max_array_index(best_params_occ)
        best_params = get_params(best_params_index, params_range)
        for p_name, p_val in best_params.items():
            f.write(f"{p_name} ; {p_val}\n")
        f.write(f"mode ; {best_params_occ[tuple(best_params_index)]}\n\n")

        f.write("-- Run details: \n")
        runs_results.to_csv(f, index=False, sep=';', line_terminator='\n')
        f.write("\n")
        f.close()

    return mean_global_stats_params, std_global_stats_params, best_params, best_params_mode


def compute_parameter_influence(model_name, model_info, dataset_name, verbose=1, location=None):
    """
    Run 10 times with 5 folds the model passed in argument on the datset passed in argument.
    All results (runs & folds, for each parameter) are saved in the folder *location*
    :param model_name:      (str) name of the model
    :param model_info:      (dict) dictionary with the information about the model (params_range, params_fct, utils_fct and function)
        parameters_range must contain only one value per parameter, except for the first one
    :param dataset_name:    (str) dataset path, datasets must be stored in data/datasets/DATASET_NAME/...
    :param verbose:         (int) verbosity for the execution
    :param location:        (str) string with the folder location for saving the results. Creates a new random folder if None
    :return:                /
    """

    ### Create the directory for saving the results if it doesn't exist yet ###
    if location is None:
        location = str(randint(1000, 9999))
    save_directory = os.path.dirname(os.path.abspath(__file__)) + '/results/' + location + '/'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    ### Initialize the parameters ###
    params_range = model_info['params_range']
    model_params = model_info['params_fct']
    model_utils = model_info['utils_fct']
    model_func = model_info['function']

    params_key = list(params_range.keys())
    n_params = len(params_key)
    keys_stats = ['train_acc', 'valid_acc', 'test_acc', 'epoch']
    grid_dim = tuple([len(val) for val in params_range.values()])

    ### Save results in a Pandas DataFrame ###
    column_names = ['run_id', 'fold_id', 'param_value'] + keys_stats
    all_results = pd.DataFrame(columns=column_names)

    varying_param_name = next(iter(params_range)) # get the parameter name
    varying_param_values = params_range[varying_param_name]
    del params_range[varying_param_name]

    params_value = dict()
    for key, value in params_range.items():
        if len(value) > 0:
            Exception(f"Parameters in 'params_range' must have only one value")
        params_value[key] = value[0]  # since there is only one value in the list

    ### Cross-validation: 10 runs of 5 folds ###
    runs = load_cross_validation_run(dataset_name)  # load all runs index associated to the given dataset
    best_params_occ = np.zeros(grid_dim)
    n_runs = len(runs)

    for i in range(n_runs):
        if verbose > 0:
            print("RUN : " + str(i))
        run = runs[i]
        t_v_fold_index = run['sub_fold']
        test_index = run['test']
        n_fold = len(t_v_fold_index)
        stats_params = [None] * n_fold
        for j in range(n_fold):
            if verbose > 0:
                print("FOLD : " + str(j))
            folds = [k for k in range(n_fold)]
            folds.remove(j)
            train_index = sum([t_v_fold_index[z] for z in folds], [])
            val_index = t_v_fold_index[j]
            stat_params = build_dict(keys_stats, grid_dim)
            for val in varying_param_values:
                params_value[varying_param_name] = val
                # fit the model
                if verbose > 0:
                    print(f"Parameters: {params_value}")
                result = fit(dataset_name, train_index, val_index, test_index, model_params, model_utils, model_func, params_value=params_value, verbose=verbose)

                all_results.loc[len(all_results)] = [str(i), str(j), str(val)] + list([el.numpy() if tf.is_tensor(el) else el for el in result.values()])

    ### Save DataFrame to file ###
    if verbose > 0:
        with pd.option_context('display.max_rows', None, 'display.max_columns', None):
            print(all_results)

    csv_name = "{}_{}.csv".format(dataset_name, model_name)
    aw = 'a' if os.path.exists(save_directory+csv_name) else 'w'

    with open(save_directory+csv_name, aw) as f:
        f.write("-- Parameters range: \n")
        f.write(f"{varying_param_name} ; {[v for v in varying_param_values]}\n")
        for p_name, p_vals in params_range.items():
            f.write(f"{p_name} ; {p_vals[0]}\n")
        f.write("\n")

        f.write("-- Run full details: \n")
        all_results.to_csv(f, index=False, sep=';', line_terminator='\n')
        f.write("\n")
        f.close()

    return all_results


def run_on_best_params(models, datasets, verbose=1, folder_path="/results/parameters_fitting/", location=None, get_param_influence=False):
    """
    This function runs each model on each dataset passed in argument and uses the best parameters previously saved in folder_path
    results are automatically saved in a .csv file
    :param models:      dictionary with all the models to run:
                        {
                            MODEL_1_NAME : {
                                'function': MODEL_1_FUNCTION_PATH,
                                'params_fct': MODEL_1_PARAMETERS_PATH,
                                'utils_fct': MODEL_1_UTILS_PATH,
                                'params_range': MODEL_1_DICT_OF_PARAMETERS_RANGES
                            },
                            MODEL_2_NAME : {...},
                            ...
                        }
    :param datasets:    list of dataset paths (strings), datasets must be stored in data/datasets/DATASET_PATH/...
    :param verbose:     verbosity for the execution
    :param folder_path: string with the relative path of the folder in which the parameters are saved
    :param location:    string with the folder location for saving the results. Creates a new random folder if None
    :return:            /
    """
    if get_param_influence:
        return_dict = dict()

    for model_name, model_info in models.items():
        if verbose > 0:
            print(f"----- Training model: {model_name}")
        for dataset in datasets:
            if verbose > 0:
                print(f"     --- Dataset: {dataset}")
            # find best params
            model_info_full = model_info.copy()
            dict_of_params = model_info["params_range"].copy()

            save_directory = os.path.dirname(os.path.abspath(__file__)) + folder_path
            csv_name = "{}_{}.csv".format(dataset, model_name)
            with open(save_directory + csv_name, 'r') as f:
                line = f.readline()
                while line.strip() != "-- Best parameters:":
                    line = f.readline()
                line = f.readline()
                while re.search(r"mode", line) is None: # add parameters
                    line = line.replace(" ", "")
                    line = line.replace("\n", "")
                    sub = line.split(";")
                    if sub[0] not in dict_of_params.keys():
                        if sub[0] != 'activation':
                            data = int(sub[1]) if sub[1].isdigit() else float(sub[1])
                            dict_of_params[sub[0]] = [data]
                        else:
                            data = str(sub[1])
                            if 'sigmoid' in data:
                                activation = tf.nn.sigmoid
                            elif 'tanh' in data:
                                activation = tf.nn.tanh
                            elif 'relu' in data:
                                activation = tf.nn.relu
                            else:
                                NameError("Activation funcion not recognized")

                            dict_of_params['activation'] = [activation]

                    line = f.readline()
                f.close()

            model_info_full["params_range"] = dict_of_params
            # run
            if get_param_influence:
                return_dict[dataset] = compute_parameter_influence(model_name, model_info_full, dataset_name=dataset, verbose=verbose, location=location)
            else:
                _,_,_,_ = validate_model(model_name, model_info_full, dataset_name=dataset, verbose=verbose, location=location)

    if get_param_influence:
        return return_dict


def run_experiments(models, datasets, verbose=1, location=None):
    """
    This function runs each model on each dataset passed in argument, results are automatically saved in a .csv file
    :param models:      dictionary with all the models to run:
                        {
                            MODEL_1_NAME : {
                                'function': MODEL_1_FUNCTION_PATH,
                                'params_fct': MODEL_1_PARAMETERS_PATH,
                                'utils_fct': MODEL_1_UTILS_PATH,
                                'params_range': MODEL_1_DICT_OF_PARAMETERS_RANGES
                            },
                            MODEL_2_NAME : {...},
                            ...
                        }
    :param datasets:    list of dataset paths (strings), datasets must be stored in data/datasets/DATASET_PATH/...
    :param verbose:     verbosity for the execution
    :param location:    string with the folder location for saving the results. Creates a new random folder if None
    :return:            /
    """

    for model_name, model_info in models.items():
        if verbose > 0:
            print(f"----- Training model: {model_name}")

        for dataset in datasets:
            if verbose > 0:
                print(f"     --- Dataset: {dataset}")
            mean_global_stats_params, std_global_stats_params, best_params, best_params_mode = validate_model(
                model_name, model_info, dataset_name=dataset, verbose=verbose, location=location)

    if verbose > 0:
        print("\nRUN COMPLETE !")



if __name__ == '__main__':
    """
    Fit and validate the main models (GCN, DCNN and GWNN) 
    using 10 runs with 5-fold-validation on multiple datasets
    """
    datasets_list = ['mytexasA100Xsym', 'mywashingtonA100Xsym', 'mywisconsinA100Xsym', 'mycornellA100Xsym',
                     'myciteseerA100Xsym', 'mycoraA100Xsym', 'myfacebookA100X107', 'myfacebookA100X1684',
                     'myfacebookA100X1912', 'mywikipediaA100Xsym', 'myamazonphotoA100Xsym', 'myamazoncomputersA100Xsym']

    models_to_run = {
        'GWNN': {
            'function': GWNN,
            'params_fct': GWNNparams,
            'utils_fct': GWNNutils,
            'params_range': {'learning_rate': [0.001, 0.01],
                             'hidden_layer_sizes': [16, 32, 64],
                             'wavelet_s': [1 / 256, 1 / 64, 1 / 16, 1 / 4, 1 / 2, 1]}
        },
        'DCNN': {
            'function': DCNN,
            'params_fct': DCNNparams,
            'utils_fct': DCNNutils,
            'params_range': {'activation': [tf.nn.tanh],
                             'learning_rate': [0.001, 0.01],
                             'hops': [1, 2, 3]}},
        'GCN': {
            'function': GCN,
            'params_fct': GCNparams,
            'utils_fct': GCNutils,
            'params_range': {'learning_rate': [0.001, 0.01],
                             'hidden_layer_sizes': [16, 32, 64]}
        },
    }

    run_experiments(models_to_run, datasets_list[4:], verbose=1, location="parameters_fitting")