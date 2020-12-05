"""
NrKmeans clustering
Implementation of the NrKmeans algorithm as described in the Paper
'Discovering Non-Redundant K-means Clusterings in Optimal Subspaces'
SubKmeans is a special case of NrKmeans if there is only a single clustering considered.
"""

import numpy as np
from scipy.stats import ortho_group
from sklearn.utils import check_random_state
from sklearn.cluster.k_means_ import _k_init as kpp
from sklearn.utils.extmath import row_norms
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.metrics import normalized_mutual_info_score as nmi

ACCEPTED_NUMERICAL_ERROR = 1e-6

"""
==================== NrKmeans Object ====================
"""


class NrKmeans():
    def __init__(self, n_clusters, V=None, m=None, P=None, centers=None, max_iter=300, allow_larger_noise_space=True,
                 random_state=None):
        """
        Create new NrKmeans instance. Gives the opportunity to use the fit() method to cluster a dataset.
        :param n_clusters: list containing number of clusters for each subspace
        :param V: orthogonal rotation matrix (optional)
        :param m: list containing number of dimensionalities for each subspace (optional)
        :param P: list containing projections for each subspace (optional)
        :param centers: list containing the cluster centers for each subspace (optional)
        :param max_iter: maximum number of iterations for the NrKmaens algorithm (default: 300)
        :param allow_larger_noise_space: if true assigns all negative eigenvalues to noise space, resulting in a larger noise space
        :param random_state: use a fixed random state to get a repeatable solution (optional)
        """
        # Fixed attributes
        self.input_n_clusters = n_clusters.copy()
        self.max_iter = max_iter
        self.random_state = check_random_state(random_state)
        # Variables
        self.n_clusters = n_clusters
        self.labels = None
        self.centers = centers
        self.V = V
        self.m = m
        self.P = P
        self.scatter_matrices = None
        self.costs = -1
        self.allow_larger_noise_space = allow_larger_noise_space
    
    def predict_subkmeans(self, X):
        return _assign_labels(X, self.V, self.centers[0], self.P[0])

    def fit(self, X, best_of_n_rounds=10, verbose=True):
        """
        Cluster the input dataset with the NrKmeans algorithm. Saves the labels, centers, V, m, P and scatter matrices
        in the NrKmeans object.
        :param X: input data
        :param best_of_n_rounds: select best result out of best_of_n_rounds
        :param verbose: if True prints the immediate results of each round
        :return: the KrKmeans object
        """
        current_best_costs = -1
        for i, seed in enumerate(self.random_state.randint(0, 10000, best_of_n_rounds)):
            labels, centers, V, m, P, n_clusters, scatter_matrices = nrkmeans(X, self.n_clusters, self.V, self.m,
                                                                              self.P,
                                                                              self.centers, self.max_iter,
                                                                              seed,
                                                                              allow_larger_noise_space=self.allow_larger_noise_space)
            this_costs = _determine_costs(scatter_matrices, P, V)
            if verbose:
                print(f"found solution with: {this_costs} (current best is {current_best_costs})")
            if this_costs < current_best_costs or current_best_costs == -1:
                current_best_labels = labels
                current_best_centers = centers
                current_best_V = V
                current_best_m = m
                current_best_P = P
                current_best_n_clusters = n_clusters
                current_best_scatter_matrices = scatter_matrices
                current_best_costs = this_costs

        self.labels = current_best_labels
        self.centers = current_best_centers
        self.V = current_best_V
        self.m = current_best_m
        self.P = current_best_P
        self.n_clusters = current_best_n_clusters
        self.scatter_matrices = current_best_scatter_matrices
        self.costs = current_best_costs
        self._rearrange_V_and_P()
        return self

    def transform_full_space(self, X):
        """
        Transform the input dataset with the orthogonal rotation matrix V from the NrKmeans object.
        :param X: input data
        :return: the rotated dataset
        """
        return np.matmul(X, self.V)

    def transform_clustered_space(self, X, subspace_index):
        """
        Transform the input dataset with the orthogonal rotation matrix V projected onto a special subspace.
        :param X: input data
        :param subspace_index: index of the subspace
        :return: the rotated dataset
        """
        cluster_space_V = self.V[:, self.P[subspace_index]]
        return np.matmul(X, cluster_space_V)

    def have_subspaces_been_lost(self):
        """
        Check weather subspaces have been lost during NrKmeans execution.
        :return: True if at least one subspace has been lost
        """
        return len(self.n_clusters) != len(self.input_n_clusters)

    def have_clusters_been_lost(self):
        """
        Check wheather clusteres within a subspace have been lost during NrKmeans execution.
        Will also return true if subspaces have been lost (check have_subspaces_been_lost())
        :return: True if at least one cluster has been lost
        """
        return not np.array_equal(self.input_n_clusters, self.n_clusters)

    def get_cluster_count_of_changed_subspaces(self):
        """
        Get the Number of clusters of the changed subspaces. If no subspace/cluster is lost, empty list will be
        returned.
        :return: list with the changed cluster count
        """
        changed_subspace = self.input_n_clusters.copy()
        for x in self.n_clusters:
            if x in changed_subspace:
                changed_subspace.remove(x)
        return changed_subspace

    def _rearrange_V_and_P(self):
        """
        Rearranges the values of V and P in such a way that the subspace-dimensions are consecutively.
        First self.m[0] columns in V belong to the first clustering
        :return:
        """
        old_V = self.V
        old_Ps = self.P

        new_Ps = []
        new_V = np.zeros(old_V.shape)

        next_free_dim = 0
        for s_i in range(len(self.P)):
            old_P = old_Ps[s_i]
            m = self.m[s_i]
            start_dim = next_free_dim
            next_free_dim = m + next_free_dim
            new_V[:, start_dim:next_free_dim] = self.V[:, old_P]
            new_Ps.append(
                np.array([i for i in range(start_dim, next_free_dim)]))

        self.P = new_Ps
        self.V = new_V


"""
==================== NrKmeans Functions ====================
"""

def nrkmeans(X, n_clusters, V, m, P, centers, max_iter, random_state, allow_larger_noise_space=True):
    """
    Execute the nrkmeans algorithm. The algorithm will search for the optimal cluster subspaces and assignments
    depending on the input number of clusters and subspaces. The number of subspaces will automatically be traced by the
    length of the input n_clusters array.
    :param X: input data
    :param n_clusters: list containing number of clusters for each subspace
    :param V: orthogonal rotation matrix
    :param m: list containing number of dimensionalities for each subspace
    :param P: list containing projections for each subspace
    :param centers: list containing the cluster centers for each subspace
    :param max_iter: maximum number of iterations for the algorithm
    :param random_state: use a fixed random state to get a repeatable solution
    :param allow_larger_noise_space: if true assigns all negative eigenvalues to noise space, resulting in a larger noise space
    :return: labels, centers, V, m, P, n_clusters (can get lost), scatter_matrices
    """
    n_clusters = n_clusters.copy()
    V, m, P, centers, random_state, subspaces, labels, scatter_matrices = _initialize_nrkmeans_parameters(
        X, n_clusters, V, m, P, centers, max_iter, random_state)
    # Check if labels stay the same (break condition)
    old_labels = None
    # Repeat actions until convergence or max_iter
    for iteration in range(max_iter):
        # Execute basic kmeans steps
        for i in range(subspaces):
            # Assign each point to closest cluster center
            labels[i] = _assign_labels(X, V, centers[i], P[i])
            # Update centers and scatter matrices depending on cluster assignments
            centers[i], scatter_matrices[i] = _update_centers_and_scatter_matrices(
                X, n_clusters[i], labels[i])
            # Remove empty clusters
            n_clusters[i], centers[i], scatter_matrices[i], labels[i] = _remove_empty_cluster(n_clusters[i], centers[i],
                                                                                              scatter_matrices[i],
                                                                                              labels[i])
        # Check if labels have not changed
        if _are_labels_equal(labels, old_labels):
            break
        else:
            old_labels = labels.copy()
        # Update rotation for each pair of subspaces
        for i in range(subspaces - 1):
            for j in range(i + 1, subspaces):
                # Do rotation calculations
                P_1_new, P_2_new, V_new = _update_rotation(
                    X, V, i, j, n_clusters, labels, P, scatter_matrices, allow_larger_noise_space=allow_larger_noise_space)
                # Update V, m, P
                m[i] = len(P_1_new)
                m[j] = len(P_2_new)
                P[i] = P_1_new
                P[j] = P_2_new
                V = V_new
        # Handle empty subspaces (no dimensionalities left) -> Should be removed
        subspaces, n_clusters, m, P, centers, labels, scatter_matrices = _remove_empty_subspace(subspaces,
                                                                                                n_clusters,
                                                                                                m, P,
                                                                                                centers,
                                                                                                labels,
                                                                                                scatter_matrices)
    # print("[NrKmeans] Converged in iteration " + str(iteration + 1))
    # Return relevant values
    return labels, centers, V, m, P, n_clusters, scatter_matrices


def _initialize_nrkmeans_parameters(X, n_clusters, V, m, P, centers, max_iter, random_state):
    """
    Initialize the input parameters form NrKmeans. This means that all input values which are None must be defined.
    Also all input parameters which are not None must be checked, if a correct execution is possible.
    :param X: input data
    :param n_clusters: list containing number of clusters for each subspace
    :param V: orthogonal rotation matrix
    :param m: list containing number of dimensionalities for each subspace
    :param P: list containing projections for each subspace
    :param centers: list containing the cluster centers for each subspace
    :param max_iter: maximum number of iterations for the algorithm
    :param random_state: use a fixed random state to get a repeatable solution
    :return: checked V, m, P, centers, random_state, number of subspaces, labels, scatter_matrices
    """
    data_dimensionality = X.shape[1]
    random_state = check_random_state(random_state)
    # Check if n_clusters is a list
    if not type(n_clusters) is list:
        raise ValueError(
            "Number of clusters must be specified for each subspace and therefore be a list.\nYour input:\n" + str(
                n_clusters))
    # Check if n_clusters contains negative values
    if len([x for x in n_clusters if x < 1]) > 0:
        raise ValueError(
            "Number of clusters must not contain negative values or 0.\nYour input:\n" + str(
                n_clusters))
    # Check if n_clusters contains more than one noise space
    nr_noise_spaces = len([x for x in n_clusters if x == 1])
    if nr_noise_spaces > 1:
        raise ValueError(
            "Only one subspace can be the noise space (number of clusters = 1).\nYour input:\n" + str(n_clusters))
    # Check if noise space is not the last member in n_clusters
    if nr_noise_spaces != 0 and n_clusters[-1] != 1:
        raise ValueError(
            "Noise space (number of clusters = 1) must be the last entry in n_clusters.\nYour input:\n" + str(
                n_clusters))
    # Get number of subspaces
    subspaces = len(n_clusters)
    # Check if V is orthogonal
    if V is None:
        V = ortho_group.rvs(dim=data_dimensionality, random_state=random_state)
    if not _is_matrix_orthogonal(V):
        raise Exception(
            "Your input matrix V is not orthogonal.\nV:\n" + str(V))
    # Calculate dimensionalities m
    if m is None and P is None:
        m = [int(data_dimensionality / subspaces)] * subspaces
        if data_dimensionality % subspaces != 0:
            choices = random_state.choice(
                range(subspaces), data_dimensionality - sum(m))
            for choice in choices:
                m[choice] += 1
    # If m is None but P is defined use P's dimensionality
    elif m is None:
        m = [len(x) for x in P]
    if not type(m) is list or not len(m) is subspaces:
        raise ValueError(
            "A dimensionality list m must be specified for each subspace.\nYour input:\n" + str(m))
    # Calculate projections P
    if P is None:
        possible_projections = list(range(data_dimensionality))
        P = []
        for dimensionality in m:
            choices = random_state.choice(
                possible_projections, dimensionality, replace=False)
            P.append(choices)
            possible_projections = list(
                set(possible_projections) - set(choices))
    if not type(P) is list or not len(P) is subspaces:
        raise ValueError(
            "Projection lists must be specified for each subspace.\nYour input:\n" + str(P))
    else:
        # Check if the length of entries in P matches values of m
        used_dimensionalities = []
        for i, dimensionality in enumerate(m):
            used_dimensionalities.extend(P[i])
            if not len(P[i]) == dimensionality:
                raise ValueError(
                    "Values for dimensionality m and length of projection list P do not match.\nDimensionality m:\n" + str(
                        dimensionality) + "\nDimensionality P:\n" + str(P[i]))
        # Check if every dimension in considered in P
        if sorted(used_dimensionalities) != list(range(data_dimensionality)):
            raise ValueError("Projections P must include all dimensionalities.\nYour used dimensionalities:\n" + str(
                used_dimensionalities))
    # Define initial cluster centers with kmeans++ for each subspace
    if centers is None:
        centers = []
        for i in range(subspaces):
            k = n_clusters[i]
            if k > 1:
                P_subspace = P[i]
                cropped_X = np.matmul(X, V[:, P_subspace])
                centers_cropped = kpp(cropped_X, k, row_norms(
                    cropped_X, squared=True), random_state)
                labels, _ = pairwise_distances_argmin_min(
                    X=cropped_X, Y=centers_cropped, metric='euclidean', metric_kwargs={'squared': True})

                centers_sub = np.zeros((k, X.shape[1]))
                # Update cluster parameters
                for center_id, _ in enumerate(centers_sub):
                    # Get points in this cluster
                    points_in_cluster = np.where(labels == center_id)[0]

                    # Update center
                    centers_sub[center_id] = np.average(
                        X[points_in_cluster], axis=0)
                centers.append(centers_sub)
            else:
                centers.append(np.expand_dims(np.average(X, axis=0), 0))

    if not type(centers) is list or not len(centers) is subspaces:
        raise ValueError(
            "Cluster centers must be specified for each subspace.\nYour input:\n" + str(centers))
    else:
        # Check if number of centers for subspaces matches value in n_clusters
        for i, subspace_centers in enumerate(centers):
            if not n_clusters[i] == len(subspace_centers):
                raise ValueError(
                    "Values for number of clusters n_clusters and number of centers do not match.\nNumber of clusters:\n" + str(
                        n_clusters[i]) + "\nNumber of centers:\n" + str(len(subspace_centers)))
    # Check max iter
    if max_iter is None or type(max_iter) is not int or max_iter <= 0:
        raise ValueError(
            "Max_iter must be an integer larger than 0. Your Max_iter:\n" + str(max_iter))
    # Initial labels and scatter matrices
    labels = [None] * subspaces
    scatter_matrices = [None] * subspaces
    return V, m, P, centers, random_state, subspaces, labels, scatter_matrices


def _assign_labels(X, V, centers_subspace, P_subspace):
    """
    Assign each point in each subspace to its nearest cluster center.
    :param X: input data
    :param V: orthogonal rotation matrix
    :param centers_subspace: cluster centers of the subspace
    :param P_subspace: projecitons of the subspace
    :return: list with cluster assignments
    """
    cropped_X = np.matmul(X, V[:, P_subspace])
    cropped_centers = np.matmul(centers_subspace, V[:, P_subspace])
    # Find nearest center
    labels, _ = pairwise_distances_argmin_min(X=cropped_X, Y=cropped_centers, metric='euclidean',
                                              metric_kwargs={'squared': True})
    # cython k-means code assumes int32 inputs
    labels = labels.astype(np.int32)
    return labels



def _update_centers_and_scatter_matrices(X, n_clusters_subspace, labels_subspace):
    """
    Update the cluster centers within this subspace depending on the labels of the data points. Also updates the
    scatter matrix of each cluster by summing up the outer product of the distance between each point and center.
    :param X: input data
    :param n_clusters_subspace: number of clusters of the subspace
    :param labels_subspace: cluster assignments of the subspace
    :return: centers, scatter_matrices - Updated cluster center and scatter matrices (one scatter matrix for each cluster)
    """
    # Create empty matrices
    centers = np.zeros((n_clusters_subspace, X.shape[1]))
    scatter_matrices = np.zeros((n_clusters_subspace, X.shape[1], X.shape[1]))
    # Update cluster parameters
    for center_id, _ in enumerate(centers):
        # Get points in this cluster
        points_in_cluster = np.where(labels_subspace == center_id)[0]
        if len(points_in_cluster) == 0:
            centers[center_id] = np.nan
            continue
        # Update center
        centers[center_id] = np.average(X[points_in_cluster], axis=0)
        # Update scatter matrix
        centered_points = X[points_in_cluster] - centers[center_id]
        for entry in centered_points:
            rank1 = np.outer(entry, entry)
            scatter_matrices[center_id] += rank1
    return centers, scatter_matrices


def _remove_empty_cluster(n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace):
    """
    Check if after label assignemnt and center update a cluster got lost. Empty clusters will be
    removed for the following rotation und iterations. Therefore all necessary lists will be updated.
    :param n_clusters_subspace: number of clusters of the subspace
    :param centers_subspace: cluster centers of the subspace
    :param scatter_matrices_subspace: scatter matrices of the subspace
    :param labels_subspace: cluster assignments of the subspace
    :return: n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace (updated)
    """
    # Check if any cluster is lost
    if np.any(np.isnan(centers_subspace)):
        # Get ids of lost clusters
        empty_clusters = np.where(
            np.any(np.isnan(centers_subspace), axis=1))[0]

        # Update necessary lists
        n_clusters_subspace -= len(empty_clusters)
        for cluster_id in reversed(empty_clusters):
            centers_subspace = np.delete(centers_subspace, cluster_id, axis=0)
            scatter_matrices_subspace = np.delete(
                scatter_matrices_subspace, cluster_id, axis=0)
            labels_subspace[labels_subspace > cluster_id] -= 1
    return n_clusters_subspace, centers_subspace, scatter_matrices_subspace, labels_subspace


def _update_rotation(X, V, first_index, second_index, n_clusters, labels, P, scatter_matrices, allow_larger_noise_space=True):
    """
    Update the rotation of the subspaces. Updates V and m and P for the input subspaces.
    :param X: input data
    :param V: orthogonal rotation matrix
    :param first_index: index of the first subspace
    :param second_index: index of the second subspace (can be noise space)
    :param n_clusters: list containing number of clusters for each subspace
    :param labels: list containing cluster assignments for each subspace
    :param P: list containing projections for each subspace
    :param scatter_matrices: list containing scatter matrices for each subspace
    :param allow_larger_noise_space: if true assigns all negative eigenvalues to noise space, resulting in a larger noise space
    :return: P_1_new, P_2_new, V_new - new P for the first subspace, new P for the second subspace and new V
    """
    # Check if second subspace is the noise space
    is_noise_space = (n_clusters[second_index] == 1)
    # Get combined projections and combined_cropped_V
    P_1 = P[first_index]
    P_2 = P[second_index]
    P_combined = np.append(P_1, P_2)
    cropped_V_combined = V[:, P_combined]
    # Prepare input for eigenvalue decomposition.
    sum_scatter_matrices_1 = np.sum(scatter_matrices[first_index], 0)
    sum_scatter_matrices_2 = np.sum(scatter_matrices[second_index], 0)
    diff_scatter_matrices = sum_scatter_matrices_1 - sum_scatter_matrices_2
    projected_diff_scatter_matrices = np.matmul(np.matmul(cropped_V_combined.transpose(), diff_scatter_matrices),
                                                cropped_V_combined)
    if not _is_matrix_symmetric(projected_diff_scatter_matrices):
        raise Exception(
            "Input for eigenvalue decomposition is not symmetric.\nInput:\n" + str(projected_diff_scatter_matrices))
    # Get eigenvalues and eigenvectors (already sorted by eigh)
    e, V_C = np.linalg.eigh(projected_diff_scatter_matrices)
    if not _is_matrix_orthogonal(V_C):
        raise Exception(
            "Eigenvectors are not orthogonal.\nEigenvectors:\n" + str(V_C))
    # Use transitions and eigenvectors to build V full
    V_F = _create_full_rotation_matrix(X.shape[1], P_combined, V_C)
    # Calculate new V
    V_new = np.matmul(V, V_F)
    if not _is_matrix_orthogonal(V_new):
        raise Exception("New V is not othogonal.\nNew V:\n" + str(V_new))

    # TODO: Check if boolean flag works as intended
    # Use number of negative eigenvalues to get new projections
    if is_noise_space:
        if allow_larger_noise_space:
            n_negative_e = len(e[e < -1e-5])
        else:
            n_negative_e = len(e[e < 0])
    else:
        n_negative_e = len(e[e < 0])

    # Update projections
    P_1_new, P_2_new = _update_projections(P_combined, n_negative_e)
    # Return new dimensionalities, projections and V
    return P_1_new, P_2_new, V_new


def _create_full_rotation_matrix(dimensionality, P_combined, V_C):
    """
    Create full rotation matrix out of the found eigenvectors. Set diagonal to 1 and overwrite columns and rows with
    indices in P_combined (consider the oder) with the values from V_C. All other values should be 0.
    :param dimensionality: dimensionality of the full rotation matrix
    :param P_combined: combined projections of the subspaces
    :param V_C: the calculated eigenvectors
    :return: the new full rotation matrix
    """
    V_F = np.identity(dimensionality)
    V_F[np.ix_(P_combined, P_combined)] = V_C
    return V_F


def _update_projections(P_combined, n_negative_e):
    """
    Create the new projections for the subspaces. First subspace gets all as many projections as there are negative
    eigenvalues. Second subspace gets all other projections in reversed order.
    :param P_combined: combined projections of the subspaces
    :param n_negative_e: number of negative eigenvalues
    :return: P_1_new, P_2_new - projections for the subspaces
    """
    P_1_new = np.array([P_combined[x] for x in range(n_negative_e)], dtype=int)
    P_2_new = np.array([P_combined[x] for x in reversed(
        range(n_negative_e, len(P_combined)))], dtype=int)
    return P_1_new, P_2_new


def _remove_empty_subspace(subspaces, n_clusters, m, P, centers, labels, scatter_matrices):
    """
    Check if after rotation and rearranging the dimensionalities a empty subspaces occurs. Empty subspaces will be
    removed for the next iteration. Therefore all necessary lists will be updated.
    :param subspaces: number of subspaces
    :param n_clusters:
    :param m: list containing number of dimensionalities for each subspace
    :param P: list containing projections for each subspace
    :param centers: list containing the cluster centers for each subspace
    :param labels: list containing cluster assignments for each subspace
    :param scatter_matrices: list containing scatter matrices for each subspace
    :return: subspaces, n_clusters, m, P, centers, labels, scatter_matrices
    """
    if 0 in m:
        np_m = np.array(m)
        empty_spaces = np.where(np_m == 0)[0]
        print(
            "[NrKmeans] ATTENTION:\nSubspaces were lost! Number of lost subspaces:\n" + str(
                len(empty_spaces)) + " out of " + str(
                len(m)))
        subspaces -= len(empty_spaces)
        n_clusters = [x for i, x in enumerate(
            n_clusters) if i not in empty_spaces]
        m = [x for i, x in enumerate(m) if i not in empty_spaces]
        P = [x for i, x in enumerate(P) if i not in empty_spaces]
        centers = [x for i, x in enumerate(centers) if i not in empty_spaces]
        labels = [x for i, x in enumerate(labels) if i not in empty_spaces]
        scatter_matrices = [x for i, x in enumerate(
            scatter_matrices) if i not in empty_spaces]
    return subspaces, n_clusters, m, P, centers, labels, scatter_matrices


def _is_matrix_orthogonal(matrix):
    """
    Check whether a matrix is orthogonal by comparing the multiplication of the matrix and its transpose and
    the identity matrix.
    :param matrix: input matrix
    :return: True if matrix is orthogonal
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    matrix_product = np.matmul(matrix, matrix.transpose())
    return np.allclose(matrix_product, np.identity(matrix.shape[0]), atol=ACCEPTED_NUMERICAL_ERROR)


def _is_matrix_symmetric(matrix):
    """
    Check whether a matrix is symmetric by comparing the matrix with its transpose.
    :param matrix: input matrix
    :return: True if matrix is symmetric
    """
    if matrix.shape[0] != matrix.shape[1]:
        return False
    return np.allclose(matrix, matrix.T, atol=ACCEPTED_NUMERICAL_ERROR)


def _are_labels_equal(labels_new, labels_old):
    """
    Check if the old labels and new labels are equal. Therefore check the nmi for each subspace. If all are 1, labels
    have not changed.
    :param labels_new: new labels list
    :param labels_old: old labels list
    :return: True if labels for all subspaces are the same
    """
    if labels_new is None or labels_old is None:
        return False
    return all([nmi(labels_new[i], labels_old[i], average_method='arithmetic') == 1 for i in range(len(labels_new))])


def _determine_costs(scatter_matrices, P, V):
    costs = 0.0
    for s_i in range(len(P)):
        cluster_space_V = V[:, P[s_i]]
        sm = np.sum(scatter_matrices[s_i], 0)
        costs += np.trace(np.matmul(np.matmul(cluster_space_V.transpose(),
                                              sm), cluster_space_V))
    return costs


