# -*- coding: utf-8 -*-
""" visualization.py

Created on 05-03-21

@author: Pierre-François De Plaen
"""

import glob
import re
import math
import itertools
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1' # run on CPU only

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import networkx as nx
import numpy as np
import scipy

import Orange

from toolbox.utils import load_dataset

pd.set_option("display.max_rows", None, "display.max_columns", None)

BASIC_COLORS = ["r","g","b","m","y","c"]
OTHER_COLORS = ["#f44336", "#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#795548", "#FFEB3B", "#607D8B", "#009688", "#CDDC39", "#00BCD4", "#E91E63", "#9E9E9E", "#FF5722"]

COLOR_MAPPING = {
    'SVM': OTHER_COLORS[12],
    'AutoSVM': OTHER_COLORS[7],
    'GCN': OTHER_COLORS[0],
    'DCNN': OTHER_COLORS[1],
    'GWNN': OTHER_COLORS[3]
}

folders_list_1 = ['parameters_fitting', 'with_dropout', 'with_feats_preprocessing', 'with_weights_regu', 'regu_L',
                  'regu_P']
folders_list_1b = ['parameters_fitting', 'with_dropout', 'with_weights_regu', 'regu_L', 'regu_P']

folders_list_2 = ['parameters_fitting', 'softmax_focal_loss', 'L1_loss', 'softmax_L1_loss', 'L2_loss',
                  'softmax_L2_loss', 'hinge_loss', 'squared_hinge_loss', 'cubed_hinge_loss']

rename_experiments_dict = {
    'parameters_fitting' : 'Baseline',
    'with_dropout' : 'Dropout',
    'with_weights_regu': 'Weights regu',
    'regu_L' : 'Regu L',
    'regu_P' : 'Regu P',
    'softmax_focal_loss' : 'sigma focal loss',
    'L1_loss' : 'L1 loss',
    'softmax_L1_loss' : 'sigma L1 loss',
    'L2_loss' : 'L2 loss',
    'softmax_L2_loss': 'sigma L2 loss',
    'hinge_loss' : 'Hinge loss',
    'squared_hinge_loss' : '(Hinge) 2 loss',
    'cubed_hinge_loss': '(Hinge) 3 loss'
}

BASE_EXTENSIONS = ("png", "svg")


def _get_structured_results(get_std=False):
    """
    Returns the results of the parameters fitting as a Pandas ('group_by') array:
                MEAN_TRAIN  MEAN_TEST   MEAN_TRAIN  MEAN_TEST    ...
                   MODEL 1    MODEL 1      MODEL 2    MODEL 2    ...
    DATASET 1   train_1,1   test_1,1    train_1,2   test_1,2     ...
    DATASET 2   train_2,1   test_2,1    train_2,2   test_2,2     ...
    DATASET 3   train_3,1   test_3,1    train_3,2   test_3,2     ...
    ...
    """
    paths = 'results/parameters_fitting/*.csv'
    files = glob.glob(paths)

    if get_std:
        full_results = pd.DataFrame(columns=["Dataset", "Num features", "Model", "mean_train", "mean_test", "std_test"])
    else:
        full_results = pd.DataFrame(columns=["Dataset", "Num features", "Model", "mean_train", "mean_test"])


    # Import from csv to pandas dataframe
    for path in files:
        with open(path, 'r') as f:
            line = f.readline()
            while line and line.strip() != "-- Run details:":
                line = f.readline()

            dataset = re.search(r"my\w+_", path)
            model = re.search(r"_M.+\.csv", "M"+path)
            if model is None:
                model = re.search(r"_\w+\.csv", "M" + path)

            content = pd.read_csv(f, sep=';')
            _, _, y = load_dataset(dataset.group()[0:-1])

            a = content.loc[content["run_id"] == "MEAN", ["train_acc", "test_acc"]]
            if get_std:
                b = content.loc[content["run_id"] == "STD", ["test_acc"]]
                full_results.loc[len(full_results)] = [dataset.group()[0:-1], y.shape[0], model.group()[1:-4],
                                                       *a["train_acc"].values, *a["test_acc"].values, *b["test_acc"].values]
            else:
                full_results.loc[len(full_results)] = [dataset.group()[0:-1], y.shape[0], model.group()[1:-4],
                                                        *a["train_acc"].values, *a["test_acc"].values]

            f.close()

    # print(full_results)
    def model_sorter(col):
        order = ['SVM', 'AutoSVM', 'MLP', 'GCN', 'DCNN', 'GWNN', 'M-GWNN', 'M-GWNN-band']
        correspondence = {model: i for i, model in enumerate(order)}
        return col.map(correspondence)

    structured_results = full_results.groupby(["Dataset", "Num features", "Model"]).sum().unstack()
    structured_results.sort_values(by=['Model'], axis='columns', key=model_sorter, inplace=True)
    # print(structured_results)

    return structured_results


def _get_results_from_folders(model_name, folders_list):
    full_results = pd.DataFrame(index=["# nodes"]+folders_list) # columns=[model_name]+folders_list

    for folder in folders_list:
        paths = 'results/'+folder+'/*.csv'
        files = glob.glob(paths)

        # Import from csv to pandas dataframe
        for path in files:
            with open(path, 'r') as f:
                line = f.readline()
                while line and "-- Run details:" not in line.strip():
                    line = f.readline()

                dataset = re.search(r"my\w+_", path)
                dataset = dataset.group()[0:-1]
                model = re.search(r"_\w+\.csv", path)
                model = model.group()[1:-4]

                if model != model_name:
                    continue

                if dataset[2:-1] not in full_results.columns:
                    full_results[dataset[2:-1]] = [0] + [0.]*len(folders_list) # add a column
                    _, _, y = load_dataset(dataset)
                    full_results.at["# nodes", dataset[2:-1]] = int(y.shape[0])

                content = pd.read_csv(f, sep=';')
                a = content.loc[content["run_id"] == "MEAN", ["test_acc"]]

                full_results.at[folder, dataset[2:-1]] = float((a["test_acc"].values)[0] * 100)

                f.close()

    return full_results


def plot_dataset_graph(dataset_name, extensions=BASE_EXTENSIONS):
    adj, _, y = load_dataset(dataset_name)
    y_class = np.argmax(y, axis=1)

    sz = 3*math.log(adj.shape[0])
    plt.figure(figsize=(sz, sz))
    ax = plt.gca()
    G = nx.from_numpy_array(adj)
    #pos = nx.kamada_kawai_layout(G)
    pos = nx.spring_layout(G)

    for c in range(y.shape[1]):
        nx.draw_networkx_nodes(
            G, pos=pos,
            nodelist=np.argwhere(y_class == c).transpose()[0,:],
            node_color=OTHER_COLORS[c],
            node_size = 120, alpha=0.8,
            label='Class '+str(c+1)
        )

    nx.draw_networkx_edges(G, pos=pos, width=1, edge_color="darkgray")
    plt.legend(scatterpoints=1, prop={'size': 22})
    plt.tight_layout()
    ax.axis("off")

    if not isinstance(extensions, list):
        extension = [extensions]

    for ex in extensions:
        file_name = "results/images/graphs/graph_{}.{}".format(dataset_name, ex)
        plt.savefig(file_name ,format=ex)
        print(f"File {file_name} saved")


def nemeny_test_and_plot(save_name="", p=0.05, extension="png"):
    if extension == "pgf":
        mpl.use("pgf")
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    structured_results = _get_structured_results()
    accuracies = structured_results.to_numpy(dtype=np.float64)
    test_accuracies = accuracies[:, 1::2]
    print(test_accuracies)

    models = ['SVM', 'AutoSVM', 'MLP', 'GCN', 'DCNN', 'GWNN']
    ranks = 1 + len(models) - scipy.stats.rankdata(test_accuracies, axis=1)
    avranks = np.mean(ranks, axis=0)
    #print(ranks)

    # Print values for Friedman and Nemenyi
    N = ranks.shape[0]  # nb datasets
    k = ranks.shape[1]  # nb models

    print(avranks)

    friedman = ( (12*N)/(k*(k+1)) ) * ( np.sum(avranks**2) - ((k*((k+1)**2))/4) )
    p_f = 1 - scipy.stats.chi2.cdf(friedman, df=(k-1))
    c_f = scipy.stats.chi2.ppf(q=1-p, df=(k-1))
    print(f"Friedman statistic value: {friedman}, with {k-1} degrees of freedom, which gives p_f={p_f}")
    print(f"    Critical value for p={p} is {c_f}")
    if p_f < p:
        print("    So we can reject null-hypothesis")
    else:
        print("    So we can NOT reject null-hypothesis")

    corrected_friedman = ((N-1)*friedman)/(N*(k-1) - friedman)
    p_c_f = 1 - scipy.stats.f.cdf(corrected_friedman, dfn=(k-1), dfd=((k-1)*(N-1)))
    c_c_f = scipy.stats.f.ppf(q=1-p, dfn=(k-1), dfd=((k-1)*(N-1)))
    print(f"Corrected Friedman is: {corrected_friedman} with {k-1} and {(k-1)*(N-1)} degrees of freedom, which gives p_c_f={p_c_f}")
    print(f"    Critical value for p={p} is {c_c_f}")
    if p_c_f < p:
        print("    So we can reject null-hypothesis")
    else:
        print("    So we can NOT reject null-hypothesis")

    cd_nemeny = Orange.evaluation.compute_CD(avranks, N, alpha=str(p))
    print(f"Critical difference for p={p} is {cd_nemeny}")
    Orange.evaluation.graph_ranks(avranks, models, cd=cd_nemeny, width=6, textspace=1.5, reverse=True)

    plt.tight_layout()
    #plt.show()
    if not isinstance(extension, list):
        extension = [extension]

    for ex in extension:
        file_name = "results/images/NEMENY_{}.{}".format(save_name, ex)
        plt.savefig(file_name ,format=ex)
        print(f"File {file_name} saved")


def bonferroni_dunn_test_and_plot(model_name, folders_list, p=0.05, comparison_type="model_improvements", extension="png"):
    if extension == "pgf":
        mpl.use("pgf")
        mpl.rcParams.update({
            "pgf.texsystem": "pdflatex",
            'font.family': 'serif',
            'text.usetex': True,
            'pgf.rcfonts': False,
        })

    improvements_results = _get_results_from_folders(model_name, folders_list)

    improvements_results.sort_values("# nodes", axis=1, inplace=True) # not mandatory...
    improvements_results.drop("# nodes", axis=0, inplace=True)

    test_accuracies = improvements_results.to_numpy(dtype=np.float64).transpose()

    if comparison_type == "model_improvements":
        experiments_names = [model_name] + [model_name+"+"+rename_experiments_dict[el] for el in folders_list[1:]]
    elif comparison_type == "losses":
        experiments_names = ["softmax C.E."] + ["".join([c if c != "_" else " " for c in el]) for el in folders_list[1:]]
    else:
        NameError(f"Invalid comparison type: {comparison_type}, please select 'model improvements' or 'losses'.")

    ranks = 1 + len(experiments_names) - scipy.stats.rankdata(test_accuracies, axis=1)
    avranks = np.mean(ranks, axis=0)

    N = ranks.shape[0]  # nb datasets
    k = ranks.shape[1]  # nb models

    # Print for Friedman

    friedman = ((12 * N) / (k * (k + 1))) * (np.sum(avranks ** 2) - ((k * ((k + 1) ** 2)) / 4))
    p_f = 1 - scipy.stats.chi2.cdf(friedman, df=(k - 1))
    c_f = scipy.stats.chi2.ppf(q=1 - p, df=(k - 1))
    print(f"Friedman statistic value: {friedman}, with {k - 1} degrees of freedom, which gives p_f={p_f}")
    print(f"    Critical value for p={p} is {c_f}")
    if p_f < p:
        print("    So we can reject null-hypothesis")
    else:
        print("    So we can NOT reject null-hypothesis")

    corrected_friedman = ((N - 1) * friedman) / (N * (k - 1) - friedman)
    p_c_f = 1 - scipy.stats.f.cdf(corrected_friedman, dfn=(k - 1), dfd=((k - 1) * (N - 1)))
    c_c_f = scipy.stats.f.ppf(q=1 - p, dfn=(k - 1), dfd=((k - 1) * (N - 1)))
    print(
        f"Corrected Friedman is: {corrected_friedman} with {k - 1} and {(k - 1) * (N - 1)} degrees of freedom, which gives p_c_f={p_c_f}")
    print(f"    Critical value for p={p} is {c_c_f}")
    if p_c_f < p:
        print("    So we can reject null-hypothesis")
    else:
        print("    So we can NOT reject null-hypothesis")

    # Print values for Bonferrony
    """c_BD = scipy.stats.truncnorm.ppf(a=0, b=+math.inf, q=(1 - (p / (k - 1))))
    print(f"difference for p={p} is {c_BD}, and CD is {c_BD * math.sqrt( (k*(k+1)) / (6*N) )}")
    for i in range(1, k):
        print(f"Comparison with {folders_list[i]}:")
        bonferroni_dunn_value = abs(avranks[0] - avranks[i]) / math.sqrt( (k*(k+1)) / (6*N) )
        p_BD = (1 - scipy.stats.truncnorm.cdf(bonferroni_dunn_value, a=0., b=+math.inf))
        print(f"    Bonferroni statistic value: {bonferroni_dunn_value}, which gives p_BD={p_BD}")
        if bonferroni_dunn_value > c_BD:
            print("    So we can reject null-hypothesis")
        else:
            print("    So we can NOT reject null-hypothesis")"""
    print("\n\n")
    #
    cd_bonferroni_dunn = Orange.evaluation.compute_CD(avranks, N, alpha=str(p), test="bonferroni-dunn")
    Orange.evaluation.graph_ranks(avranks, experiments_names, cd=cd_bonferroni_dunn, cdmethod=0, width=5.5+2*0.3, textspace=1.8, reverse=True)

    plt.tight_layout()
    #plt.show()
    if not isinstance(extension, list):
        extension = [extension]

    for ex in extension:
        file_name = "results/images/bonferroni/BONFERRONI-DUNN_{}_{}_{}.{}".format(model_name, comparison_type, str(p), ex)
        plt.savefig(file_name, format=ex)
        print(f"File {file_name} saved")


def get_datasets_stats():
    for dataset_name in datasets_list:
        A, X, y = load_dataset(dataset_name)
        print(f"dataset {dataset_name}: \n"
              f"A.shape = {A.shape} \n "
              f"X.shape = {X.shape} \n"
              f"y.shape = {y.shape} \n"
              f"Tot num edges = {np.count_nonzero(A) // 2} \n"
              f"Sum of diagonal = {np.sum(np.diagonal(A))} \n"
              f"Sparsity of the features: {100*np.count_nonzero(X)/(X.shape[0]*X.shape[1])}")


if __name__ == "__main__":
    datasets_list = ['myciteseerA100Xsym', 'mycoraA100Xsym', 'mycornellA100Xsym', 'myfacebookA100X107', 'myfacebookA100X1684', 'myfacebookA100X1912', 'mytexasA100Xsym', 'mywashingtonA100Xsym', 'mywisconsinA100Xsym', 'mywikipediaA100Xsym', 'myamazonphotoA100Xsym', 'myamazoncomputersA100Xsym']

    for dataset in datasets_list:
        plot_dataset_graph(dataset, extensions=("svg", "png"))



