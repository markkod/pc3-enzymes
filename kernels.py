import torch
import os
import numpy as np
import kernel_baselines as kb
import auxiliarymethods
from auxiliarymethods import datasets as dp
from scipy.sparse import save_npz
from scipy.sparse import load_npz
from scipy.sparse import csr_matrix
from auxiliarymethods import auxiliary_methods as aux
from auxiliarymethods import kernel_evaluation as ke
from auxiliarymethods.reader import tud_to_networkx
from matplotlib import pyplot as plt
import networkx as nx
from collections import defaultdict
import pandas as pd
from install import install_dependencies

def setup_directory(dir_name, verbose=False):
    """Setup directory in case it does not exist.

    Args:
        dir_name ([str]): path + name to directory
        verbose (bool, optional): Indicates whether directory creation should be printed or not. Defaults to False.

    Raises:
        RuntimeError:
    """
    if not os.path.exists(dir_name):
        try:
            os.makedirs(dir_name)
            if verbose:
                print("Created Directory: {}".format(dir_name))
        except Exception as e:
            raise RuntimeError(
                "Could not create directory: {}\n {}".format(dir_name, e))


def load_csv(path):
    """Loads a csv file from a given path.

    Args:
        path (string): path

    Returns:
        ndarray: Data read from the text file.
    """
    return np.loadtxt(path, delimiter=";")


def load_sparse(path):
    """Loads a sparse matrix from a given npz file.

    Args:
        path (string): path to .npz file

    Returns:
        csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix: A sparse matrix containing the loaded data.
    """
    return load_npz(path)


def visualize(G, color=None, figsize=(5, 5)):
    """Visualizes the given graph.

    Args:
        G (Graph): graph to be visualized
        color ([type], optional): color of the graph. Defaults to None.
        figsize (tuple, optional): Visualization size. Defaults to (5, 5).
    """
    plt.figure(figsize=figsize)
    plt.xticks([])
    plt.yticks([])
    nx.draw_networkx(G,
                     pos=nx.spring_layout(G, seed=42),
                     with_labels=True,
                     node_color=color,
                     cmap="Set2")
    plt.show()


def graph_list_properties(graphs, prop):
    values = []
    for g in graphs:
        values.append(prop(g))
    return values


def find_keys_with_condition(data, cond):
    """Finds a key that contains the given condition.

    Args:
        data (dict): Dictionary to be searched
        cond (string): Condition string

    Returns:
        list: keys that match the condition
    """
    return list(filter(lambda x: cond in x, data.keys()))


def load_data():
    """Loads the ENZYMES dataset.

    Returns:
        dict: dataset
    """
    result = {}
    extensions = ['csv', 'npz']
    types = ['gram_matrix', 'vectors']
    algos = ['wl1', 'wl2', 'wl3', 'wl4', 'wl5', 'wl6',
             'wl7', 'wl8', 'shortestpath', 'graphlet']
    base_name = '/content/tudataset/tud_benchmark/kernels/node_labels/ENZYMES_{0}_{1}.{2}'

    for t, e in zip(types, extensions):
        result[t] = {}
        for a in algos:
            algo_name = 'wl' if 'wl' in a else a
            if algo_name not in result[t].keys():
                result[t][algo_name] = []
            file_name = base_name.format(t, a, e)
            if e == 'csv':
                f = np.loadtxt(file_name, delimiter=';')
            else:
                f = load_npz(file_name)
            result[t][algo_name].append(f)
    return result


def eval_kernel(kernel, classes, mode, n_reps=10, all_std=True):
    """Evaluates a specific kernel that will be normalized before evaluation.

    Args:
        kernel ([list]): kernel
        classes (list): dataset classes
        mode (string): either LINEAR or KERNEL
        n_reps (int, optional): Number of repetitions. Defaults to 10.
        all_std (bool, optional): Standard deviation?. Defaults to True.

    Returns:
        tuple: evaluation results
    """
    normalized = []
    print(f'Starting normalization of {len(kernel)} elements...')
    for array in kernel:
        if mode == 'LINEAR':
            normalized.append(aux.normalize_feature_vector(array))
        else:
            normalized.append(aux.normalize_gram_matrix(array))
    print(f'Normalization finished, starting {mode} SVM...')
    if mode == 'LINEAR':
        return ke.linear_svm_evaluation(normalized, classes, num_repetitions=n_reps, all_std=all_std)
    return ke.kernel_svm_evaluation(normalized, classes, num_repetitions=n_reps, all_std=all_std)


def eval_all(data):
    """Evaluates the kernels on the data.

    Args:
        data ([type]): [description]

    Returns:
        [type]: [description]
    """
    classes = dp.get_dataset('ENZYMES')
    result = {}
    for data_type in data.keys():
        mode = 'LINEAR' if data_type == 'vectors' else 'KERNEL'
        result[data_type] = {}
        print('MODE:', mode)
        for kernel in data[data_type]:
            print(f'\nEvaluating {kernel} SVM...')
            result[data_type][kernel] = eval_kernel(
                data[data_type][kernel], classes, mode)
            print(f'{data_type}-{kernel} : {result[data_type][kernel]}')
    return result


def get_labels(G):
    """Gets all the various labels of a graph

    Args:
        G (Graph): graph

    Returns:
        [set]: Set of labels
    """
    labels = set()
    for g in G:
        for i in range(g.number_of_nodes()):
            labels.add(g.nodes[i]["labels"][0])
    return labels


def get_graph_dict(G, classes):
    """Get a dictionary with classes and nodes.

    Args:
        G (graph): graph
        classes (classes): classes

    Returns:
        [dict]: classes and their nodes
    """
    graph_dict = defaultdict(list)
    for i, g in enumerate(G):
        graph_dict[classes[i]].append(g)
    return graph_dict


def print_graph_information(graph_dict):
    """Prints the information about the graph and its classes.

    Args:
        graph_dict (dict): classes and its nodes
    """
    for label_class in graph_dict:
        print_class_nr_nodes(graph_dict, label_class)
        print_class_cliques(graph_dict, label_class)


def print_class_nr_nodes(graph_dict, label_class):
    """Prints the number of nodes in a single class.

    Args:
        graph_dict (dict): graph classes -> nodes
        label_class (int): label class
    """
    print("Num nodes:")
    num_nodes = graph_list_properties(
        graph_dict[label_class], lambda g: g.number_of_nodes())
    print(label_class, num_nodes)


def print_class_cliques(graph_dict, label_class):
    """Prints the cliques of nodes in a single class.

    Args:
        graph_dict (dict): graph classes -> nodes
        label_class (int): label class
    """
    print("Cliques:")
    num_nodes = graph_list_properties(
        graph_dict[label_class], lambda g: nx.algorithms.wiener_index(g))
    print(label_class, np.average(
        np.array([n for n in num_nodes if str(n) != "inf"])), num_nodes)


def eval_wl(data, classes):
    """Evaluates the gram matrices of WL kernels.

    Args:
        data (list): data
        classes ([list]): classes
    """
    for array in data["gram_matrix"]["wl"]:
        normalized = [aux.normalize_gram_matrix(array)]
        print(ke.kernel_svm_evaluation(normalized,
                                       classes, num_repetitions=10, all_std=True))


def run(with_install=True):
    if with_install:
        install_dependencies()
    base_path = os.path.join("kernels", "node_labels")
    ds_name = "ENZYMES"
    classes = dp.get_dataset(ds_name)
    G = tud_to_networkx(ds_name)
    print(f"Number of graphs in data set is {len(G)}")
    print(f"Number of classes {len(set(classes.tolist()))}")

    labels = get_labels(G)
    graph_dict = get_graph_dict(G, classes)

    print_graph_information(graph_dict)

    visualize(graph_dict[6][7])
    graph_dict[6][7].number_of_nodes()
    data = load_data()

    eval_wl(data, classes)

    max_nodes = max(map(lambda x: x.number_of_nodes(), G))
    histograms = csr_matrix((len(G), max_nodes))
    for i, g in enumerate(G):
        for n, d in g.degree():
            histograms[i, n] = d

    histogram_gram = histograms @ histograms.T

    centrality = csr_matrix((len(G), max_nodes))
    for i, g in enumerate(G):
        for n, d in nx.degree_centrality(g).items():
            centrality[i, n] = d

    centrality_gram = centrality @ centrality.T
    val = data["vectors"]["wl"][2].T.dot(histograms)
    val = data["vectors"]["wl"][2].T.dot(histograms)
    print(val.shape)
    normalized = [aux.normalize_feature_vector(val)]
    print(normalized[0].shape)
    print(ke.linear_svm_evaluation(normalized,
                                   classes, num_repetitions=10, all_std=True))
