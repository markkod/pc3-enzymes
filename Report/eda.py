from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi

def plot_wl_nmi_comparison():
    algorithms = ["KPCA", "TSVD", "SPEC"]
    for version in ["node_labels", "without_labels"]:
        print(f"#################{version}#################")
        all_nmi = {}

        for algorithm in algorithms:
            path_i = os.path.join('kernels', version, f'{algorithm}.csv')
            nmi_df = pd.read_csv(path_i, index_col=0)
            all_nmi[algorithm] = nmi_df['ENZYMES']

            max_nmi = nmi_df['ENZYMES'].max()
            max_nmi_id = nmi_df['ENZYMES'].idxmax()

        all_nmi_df = pd.DataFrame.from_dict(all_nmi)from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.decomposition import KernelPCA, TruncatedSVD
from scipy.sparse import load_npz
import auxiliarymethods.auxiliary_methods as aux
import os
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy as np

def load_csv(path):
    return np.loadtxt(path, delimiter=";")

def load_sparse(path):
    return load_npz(path)

def plot_wl_nmi_comparison():
  algorithms = ["KPCA", "TSVD", "SPEC"]
  for version in ["node_labels", "without_labels"]:
    print(f"#################{version}#################")
    all_nmi = {}

    for algorithm in algorithms:
        path_i = os.path.join('kernels', version, f'{algorithm}.csv')
        nmi_df = pd.read_csv(path_i, index_col=0)
        all_nmi[algorithm] = nmi_df['ENZYMES']

        max_nmi = nmi_df['ENZYMES'].max()
        max_nmi_id = nmi_df['ENZYMES'].idxmax()

    all_nmi_df = pd.DataFrame.from_dict(all_nmi)

    fig, ax = plt.subplots(figsize=(10,3))
    ax.set_ylabel("NMI")
    ax.set_xlabel("WL-Iterations")
    ax.set_xticks([0,1,2,3,4,5,6])
    ax.set_xticklabels([1,2,3,4,5, 'graphlet', 'shortestpath'])
    ax.set_title(version)
    all_nmi_df.plot(marker="o", ax=ax)
    plt.show()


def plot_representations_nmi_comparison():
  rc = pd.read_csv('representation_comparison.csv')
  rc_node_labels = rc[rc.Labels == True]
  rc_without_labels = rc[rc.Labels == False]

  fig, axs = plt.subplots(1,3, figsize=(15,4))

  for (i, algorithm) in enumerate(rc.Algorithm.unique()):
    df = rc[rc.Algorithm==algorithm]
    axs[i].set_ylabel("NMI")
    axs[i].set_xlabel("Representation")
    axs[i].set_title(algorithm)
    axs[i].set_ylim([0,0.1])

    sns.barplot(x=df['Representation'], y=df["NMI"], data=df, hue='Labels', ax=axs[i])

  plt.show()

def plot_kpca_nmi_and_clustering(classes):
  ds_name = 'ENZYMES'
  base_path = os.path.join("kernels","node_labels")

  fig, axs = plt.subplots(3,3, figsize=(15,15))
  representations = ["wl3", "graphlet", "shortestpath"]

  for (i, representation) in enumerate(representations): 
    gram = load_csv(os.path.join(base_path,f"{ds_name}_gram_matrix_{representation}.csv"))
    gram = aux.normalize_gram_matrix(gram)

    kpca = KernelPCA(n_components=100, kernel="precomputed")
    reduced_kpca = kpca.fit_transform(gram)
    # fig, ax = plt.subplots(figsize=(5,5))
    axs[0][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=classes, s=1)
    axs[0][i].set_title(f'{representation} KPCA ground truth')
    
    kmeans = KMeans(n_clusters=6).fit(reduced_kpca)
    axs[1][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=kmeans.labels_, s=1)
    axs[1][i].set_title(f'{representation} KPCA KMeans')
    print(f"NMI KMeans {representation}: {nmi(classes, kmeans.labels_)}")

    db = DBSCAN().fit(reduced_kpca)
    axs[2][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=db.labels_, s=1)
    axs[2][i].set_title(f'{representation} DBSCAN KMeans')
    print(f"NMI DBSCAN {representation}: {nmi(classes, db.labels_)}\n")

  plt.show()

def plot_tsvd_nmi_and_clustering(classes):
  ds_name = 'ENZYMES'
  base_path = os.path.join("kernels","node_labels")

  fig, axs = plt.subplots(3,3, figsize=(15,15))
  representations = ["wl3", "graphlet", "shortestpath"]

  for (i, representation) in enumerate(representations): 
      vec = load_sparse(os.path.join(base_path,f"{ds_name}_vectors_{representation}.npz"))
      tsvd = TruncatedSVD(n_components=100)
      reduced_tsvd = tsvd.fit_transform(vec)

      # fig, ax = plt.subplots(figsize=(5,5))
      axs[0][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=classes, s=1)
      axs[0][i].set_title("Representation: " + representation)
      axs[0][i].set_title(f'{representation} TSVD')

      kmeans = KMeans(n_clusters=6).fit(reduced_tsvd)
      axs[1][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=kmeans.labels_, s=1)
      axs[1][i].set_title(f'{representation} TSVD KMeans')
      print(f"NMI KMeans {representation}: {nmi(classes, kmeans.labels_)}")

      db = DBSCAN().fit(reduced_tsvd)
      axs[2][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=db.labels_, s=1)
      axs[2][i].set_title(f'{representation} DBSCAN KMeans')
      print(f"NMI DBSCAN {representation}: {nmi(classes, db.labels_)}\n")
  
  plt.show()

        fig, ax = plt.subplots(figsize=(10,3))
        ax.set_ylabel("NMI")
        ax.set_xlabel("WL-Iterations")
        ax.set_xticks([0,1,2,3,4,5,6])
        ax.set_xticklabels([1,2,3,4,5, 'graphlet', 'shortestpath'])
        ax.set_title(version)
        all_nmi_df.plot(marker="o", ax=ax)
        plt.show()


def plot_representations_nmi_comparison():
    rc = pd.read_csv('representation_comparison.csv')
    rc_node_labels = rc[rc.Labels == True]
    rc_without_labels = rc[rc.Labels == False]

    fig, axs = plt.subplots(1,3, figsize=(15,4))

    for (i, algorithm) in enumerate(rc.Algorithm.unique()):
    df = rc[rc.Algorithm==algorithm]
    axs[i].set_ylabel("NMI")
    axs[i].set_xlabel("Representation")
    axs[i].set_title(algorithm)
    axs[i].set_ylim([0,0.1])

    sns.barplot(x=df['Representation'], y=df["NMI"], data=df, hue='Labels', ax=axs[i])

    plt.show()

def plot_kpca_nmi_and_clustering():
    ds_name = 'ENZYMES'

    fig, axs = plt.subplots(3,3, figsize=(15,15))
    representations = ["wl3", "graphlet", "shortestpath"]

    for (i, representation) in enumerate(representations): 
        gram = load_csv(os.path.join(base_path,f"{ds_name}_gram_matrix_{representation}.csv"))
        gram = aux.normalize_gram_matrix(gram)

        kpca = KernelPCA(n_components=100, kernel="precomputed")
        reduced_kpca = kpca.fit_transform(gram)
        # fig, ax = plt.subplots(figsize=(5,5))
        axs[0][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=classes, s=1)
        axs[0][i].set_title(f'{representation} KPCA ground truth')
        
        kmeans = KMeans(n_clusters=6).fit(reduced_kpca)
        axs[1][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=kmeans.labels_, s=1)
        axs[1][i].set_title(f'{representation} KPCA KMeans')
        print(f"NMI KMeans {representation}: {nmi(classes, kmeans.labels_)}")

        db = DBSCAN().fit(reduced_kpca)
        axs[2][i].scatter(reduced_kpca[:,0], reduced_kpca[:,1], c=db.labels_, s=1)
        axs[2][i].set_title(f'{representation} DBSCAN KMeans')
        print(f"NMI DBSCAN {representation}: {nmi(classes, db.labels_)}\n")

    plt.show()

def plot_tsvd_nmi_and_clustering():
    ds_name = 'ENZYMES'

    fig, axs = plt.subplots(3,3, figsize=(15,15))
    representations = ["wl3", "graphlet", "shortestpath"]

for (i, representation) in enumerate(representations): 
    vec = load_sparse(os.path.join(base_path,f"{ds_name}_vectors_{representation}.npz"))
    tsvd = TruncatedSVD(n_components=100)
    reduced_tsvd = tsvd.fit_transform(vec)

    # fig, ax = plt.subplots(figsize=(5,5))
    axs[0][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=classes, s=1)
    axs[0][i].set_title("Representation: " + representation)
    axs[0][i].set_title(f'{representation} TSVD')

    kmeans = KMeans(n_clusters=6).fit(reduced_tsvd)
    axs[1][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=kmeans.labels_, s=1)
    axs[1][i].set_title(f'{representation} TSVD KMeans')
    print(f"NMI KMeans {representation}: {nmi(classes, kmeans.labels_)}")

    db = DBSCAN().fit(reduced_tsvd)
    axs[2][i].scatter(reduced_tsvd[:,0], reduced_tsvd[:,1], c=db.labels_, s=1)
    axs[2][i].set_title(f'{representation} DBSCAN KMeans')
    print(f"NMI DBSCAN {representation}: {nmi(classes, db.labels_)}\n")
    plt.show()