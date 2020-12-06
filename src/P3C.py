from scipy.stats import chisquare,chi2
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy as sp
import sys
from sklearn import mixture, preprocessing
from scipy.stats import poisson
import csv
from collections import defaultdict
from pprint import pprint

from src.bin import Bin
from src.interval import Interval
from src.psignature import PSignature
from src.datapoint import DataPoint

    
def load_data(path, delimeter=','):
    data = []
    with open(path, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=delimeter, quotechar='|')
        for row in reader:
            if len(row) > 0 and not row[0].startswith('#'):
                data.append([float(r) for r in row])
    return data
                
                
def normalize_data(data):
    scaler = preprocessing.MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data


def split_into_bins(data):
    bins = []
    for col_idx in range(len(data[0])):
        column_bins = []
        column_data = [x[col_idx] for x in data]
        col_min = min(column_data)
        col_max = max(column_data)

        interval_length = col_max - col_min
        # divide interval 1 + log2(n) bins
        nr_of_bins = math.floor(1 + math.log2(len(column_data)))
        
        column_bins = []
        b = interval_length / nr_of_bins
        for i in range(nr_of_bins):
            # adds a bin with a start and end interval
            bin_interval = Interval(col_min + b * i, col_min + b * (i + 1))
            
            column_bins.append(Bin(i, bin_interval, col_idx))

        # compute support for each bin
        for i, datapoint in enumerate(column_data):
            bin_index = int((datapoint - col_min) / b)
            bin_index = bin_index if bin_index < nr_of_bins else bin_index - 1
            column_bins[bin_index].add_point(data[i])

        bins.append(column_bins)
    return bins, nr_of_bins


# find the bin with the highest support and mark it
def mark_highest_support(column_bins):
    max_support = 0
    max_index = 0
    for _bin in column_bins:
        if _bin.marked:
            continue
        if _bin.support > max_support:
            max_support = _bin.support
            max_index = _bin.index
    column_bins[max_index].marked = True


# perform chisquared for the support 
def mark_bins(column_bins, nr_of_bins, alpha=0.001, stat=1):
    while (stat > alpha):
        # support list of all *unmarked* bins
        support_list = [column_bins[i].support for i in range(nr_of_bins) if not column_bins[i].marked]
        # print(support_list)
        # if there are no unmarked bins, end the process
        if len(support_list) == 0: 
            break
        (stat, p) = chisquare(support_list)
        #print('stat', stat)
        #print('p', p)
        if (stat > alpha):
            mark_highest_support(column_bins)
    return column_bins


def mark_merge_bins(column_bins):
    for i, _bin1 in enumerate(column_bins):
        for j, _bin2 in enumerate(column_bins[i+1:]):
            if _bin1.marked == _bin2.marked:
                _bin1.merge_with.append(_bin2.index)
            else:
                break


# merge each bin in the list by extending the interval and combining support
def merge_bin(all_column_bins, column_bin):
    for bin_index in column_bin.merge_with:
            column_bin.interval.end = all_column_bins[bin_index].interval.end
            column_bin.support += all_column_bins[bin_index].support
            column_bin.assigned_points.extend(all_column_bins[bin_index].assigned_points)


# merge bins of a single column
def merge_column_bins(column_bins):
    i = 0
    new_bins = []
    while i < len(column_bins):
        # if bin has no following bins to merge with, keep it as is, and go to the next one
        if len(column_bins[i].merge_with) == 0:
            new_bins.append(column_bins[i])
            i += 1
            continue
            
        merge_bin(column_bins, column_bins[i])
        
        new_bins.append(column_bins[i])
        
        # skip all of the bins that were included in the current one
        i = max(column_bins[i].merge_with) + 1
        
    return new_bins


def merge_all_bins(bins):
    new_bins = []
    for column_bins in bins:
        new_bins.append(merge_column_bins(column_bins))
    return new_bins


def create_new_candidate(candidate, dim_bin, reevaluated_points):
    current_bins_list = []
    current_bins_list.extend(candidate.bins)
    current_bins_list.append(dim_bin)
    return PSignature(current_bins_list, reevaluated_points)


# def generate_candidate_list(candidate_list, current_dim_bins, threshold, current_dim):
#     new_candidates = []
#     for candidate in candidate_list:
#         for dim_bin in current_dim_bins:
#             if dim_bin.marked:
#                 expected_sup = candidate.get_support() * dim_bin.get_width()            
#                 reevaluated_points = candidate.reevaluate_assigned_points(dim_bin, current_dim)
#                 r_support = len(reevaluated_points)
#                 if r_support == 0:
#                     continue
#                 print('R support {0}, expected support {1}'.format(r_support, expected_sup))
#                 print('Poisson distribution:', poisson.pmf(r_support, expected_sup), r_support, expected_sup)

#                 if poisson.pmf(r_support, expected_sup) < threshold:
#                     new_candidate = create_new_candidate(candidate, dim_bin, reevaluated_points)
#                     new_candidates.append(new_candidate)
#                     print("Length of new candidates after poisson", len(new_candidates))
#     return new_candidates


def construct_candidate_tree_start(data, new_bins):
    candidate_tree = {}
    candidate_tree_0 = {}
    for dim in range(0, len(data[0])):
        candidate_dict = {}
        for _bin in new_bins[dim]:
            if _bin.marked:
                candidate_dict[f'{_bin.id}'] = PSignature([_bin], assigned_points=_bin.assigned_points)
        dim_id = 'd{}'.format(dim)
        candidate_tree_0[dim_id] = candidate_dict
    candidate_tree[0] = candidate_tree_0
    return candidate_tree


def construct_new_level(parent_level_id, candidate_tree, threshold):
    stop = False
    print("Constructing candidate tree: ")
    while not stop:
        parent_level = candidate_tree[parent_level_id]
        current_candidates_ids = []
        new_level_tree = {}
        print("    Level: ", parent_level_id)
        for (p1_id, p1_psigs) in parent_level.items():
            #print("parent_level items", p1_id)
            for (p2_id, p2_psigs) in parent_level.items():
                l = len(p1_id)
                p1_ids = p1_id.split()
                p2_ids = p2_id.split()
                if (' '.join(p1_ids[0:-1]) == ' '.join(p2_ids[0:-1]) and p1_ids[-1] != p2_ids[-1]):
                    current_candidate_id_list = [p2_ids[-1]]
                    current_candidate_id_list += p1_id.split(' ')
                    current_candidate_id = ' '.join(sorted(current_candidate_id_list))
                    if current_candidate_id not in current_candidates_ids:
                        current_candidates_ids.append(current_candidate_id)
                        res = construct_new_signatures(p1_psigs, p2_psigs, threshold)
                        new_level_tree[current_candidate_id] = res
        if not new_level_tree:
            stop = True
        candidate_tree[parent_level_id + 1] = new_level_tree
        parent_level_id += 1
    return candidate_tree
                    

def construct_new_signatures(p1_psigs, p2_psigs, threshold):
    new_candidates = {}
    for _, p1_psig in p1_psigs.items():
        p1_psig_id = ' '.join(sorted(p1_psig.id))
        l = len(p1_psig_id)
        for _, p2_psig in p2_psigs.items():
            p2_psig_id = ' '.join(sorted(p2_psig.id))
            if (set(p1_psig.id[0:-1]) == set(p2_psig.id[0:-1])):
                dim_bin_id = p2_psig.id[-1]
                p12_psig_id = ' '.join(sorted([p1_psig_id, dim_bin_id]))
                
                dim_bin = p2_psig.bin_dict[dim_bin_id]
                if dim_bin.marked:
                    candidate = p1_psig
                    expected_sup = candidate.get_support() * dim_bin.get_width()            
                    reevaluated_points = candidate.reevaluate_assigned_points(dim_bin, dim_bin.dimension)
                    r_support = len(reevaluated_points)
                    if r_support == 0:
                        continue
#                     print('R support {0}, expected support {1}'.format(r_support, expected_sup))
#                     print('Poisson distribution:', poisson.pmf(r_support, expected_sup), r_support, expected_sup)

                    if poisson.pmf(r_support, expected_sup) < threshold:
                        p1_psig.parent = True
                        p2_psig.parent = True
                        new_candidate = create_new_candidate(candidate, dim_bin, reevaluated_points)
                        new_candidates[p12_psig_id] = new_candidate
#                         print('{} + {} = {}'.format(p1_psig_id, p2_psig_id, p12_psig_id))
#                         print("Length of new candidates after poisson", len(new_candidates))

    return new_candidates


def get_candidates(tree):
    candidates = {}
    for (level, dim_candidates) in tree.items():
        for (dim_candidate_id, dim_bins_candidates) in dim_candidates.items():
            for (dim_bins_candidate_id, dim_bins_candidate) in dim_bins_candidates.items():
                if not dim_bins_candidate.parent and len(dim_bins_candidate.bins) > 1:
                    candidates[dim_bins_candidate_id] = dim_bins_candidate

    for (c1_id, c1) in candidates.items():
        c1_set = set(c1_id.split(' '))
        for (c2_id, c2) in candidates.items():
            if c1_id != c2_id and c1_set.issubset(set(c2_id.split(' '))):
                c1.parent=True
                break

    candidates = [c for _, c in candidates.items() if not c.parent]
    candidates
    candidate_list = candidates
    return candidate_list


def test_thresholds():
    for t in range(1, 20):
        threshold = 1e-2 / (10**t)
        tree = construct_candidate_tree_start()
        ns = construct_new_level(0, tree, threshold)
        candidates = get_candidates(ns)
        print(f"For threshold: {threshold} found {len(candidates)} candidates")


def get_inv_cov_cluster_dict(candidate_list):
    inv_cov_cluster_dict = dict()

    for i,can in enumerate(candidate_list):   
        cov = np.cov(np.array(can.assigned_points).T)
        inv_covmat= np.linalg.inv(cov)
        inv_cov_cluster_dict[i] = inv_covmat
    
    return inv_cov_cluster_dict


def get_result(data, candidate_list, inv_cov_cluster_dict):
    #fuzzy membership matrix
    #initialize matrix with datapoints in one column and found cluster (e.g. 1,2,3) in other column
    #initialize clusterpoints with a 1 at the matrix intersection
    matrix = np.zeros(dtype='float', shape=(len(data), len(candidate_list)))
    dps = []

    # print(matrix.shape)

    cov_dat = np.cov(np.array(data).T)
    inv_covmat_dat= np.linalg.inv(cov_dat)

    print(f"Constructing fuzzy matrix for {len(data)} datapoints: ")
    for i, point in enumerate(data):
        if i % 100 == 0:
            print(f"    {i}/{len(data)}")
        data_point = DataPoint(point)
        for j, candidate in enumerate(candidate_list):
            candidate_data_points = [DataPoint(p) for p in candidate.assigned_points]
            if data_point in candidate_data_points:
                matrix[i][j] = 1
                data_point.assigned_clusters.append(j)
        fraction = 1 if len(data_point.assigned_clusters) == 0 else 1 / len(data_point.assigned_clusters) 
        for r in range(len(candidate_list)):
            if matrix[i][r] == 1:
                matrix[i][r] = fraction
        #"""
        if len(data_point.assigned_clusters) == 0:
            closest = sys.maxsize
            closest_candidate_idx = 0
            for idx, c in enumerate(candidate_list):
                mh_distance = distance.mahalanobis(data_point.coords, c.get_means(), inv_cov_cluster_dict[idx])
                if mh_distance < closest:
                    closest = mh_distance
                    closest_candidate_idx = idx
            data_point.assigned_clusters.append(closest_candidate_idx)
            matrix[i][closest_candidate_idx] = 1
        #"""
        dps.append(data_point)
                    
    #compute mean of support set of cluster

    #compute the shortest mahalanobis distance(scipy.spatial.distance.mahalanobis) 
    # of unassigned points to cluster core and assign

    # EM -> probably need to implement ourself
    means = np.array([c.get_means() for c in candidate_list])

    gmm = mixture.BayesianGaussianMixture(n_components=len(candidate_list), covariance_type='full').fit(matrix)

    # gmm = mixture.GaussianMixture(n_components=len(candidate_list), covariance_type='full').fit(data)

    result = gmm.predict(matrix)

    # result = gmm.predict(data)

    return result, gmm, means


def get_clusters_and_means(candidate_list, data, result, means_before):
    clustered_points = list()
    projected_cluster_dict = defaultdict(list)

    for assigned_cluster, p in list(zip(result, data)):
        clustered_points.append((assigned_cluster,p))
        projected_cluster_dict[assigned_cluster].append(p)

    means_after_bgm = {}       
    for pj in projected_cluster_dict.keys():
        if len(projected_cluster_dict[pj]) == 0:
            continue
            mean = np.zeros(len(data[0]))
        else:
            mean = np.mean(np.array(projected_cluster_dict[pj]), axis = 0)
        means_after_bgm[pj] = mean

    amount = 0    
    for pj in projected_cluster_dict.keys():
        amount += len(projected_cluster_dict[pj])

    # print(amount)
    print("Final cluster means:")
    pprint(means_after_bgm)
    return means_after_bgm, projected_cluster_dict, clustered_points


def plot_means(data, means_before, means_after_bgm, result):
    plt.scatter([x[0] for x in data], [y[1] for y in data], c=result, s=20)
    plt.scatter([x[0] for _, x in means_after_bgm.items()], [y[1] for _, y in means_after_bgm.items()], c="green")
    plt.scatter([x[0] for x in means_before], [y[1] for y in means_before], c="red")

    plt.show()


def find_outliers(data, candidate_list, projected_cluster_dict, clustered_points, means_after_bgm, result, degree_of_freedom=None, alpha=0.001):
    inv_cov_dict = dict()

    for key in projected_cluster_dict.keys():  
        cov = np.cov(np.array(projected_cluster_dict[key]).T)
        try:
            inv_covmat = np.linalg.inv(cov)
        except:
            inv_covmat = np.zeros(cov.shape)
        inv_cov_dict[key] = inv_covmat
        
    if not degree_of_freedom:
        degree_of_freedom = len(set(result)) ** 2
    #degree_of_freedom = 10
    chi_crit = chi2.ppf(alpha, df=degree_of_freedom)

    noise_cluster_idx  = len(candidate_list)
    print("Chi critical value for outline detection: ", chi_crit)
    for i, c in enumerate(clustered_points):
        cluster_mean = means_after_bgm[c[0]] 
        md = distance.mahalanobis(c[1],cluster_mean,inv_cov_dict[c[0]])
        if md > chi_crit:
            clustered_points[i] = (noise_cluster_idx,c[1])
            print(f"    Found an outlier with the distance {md} at point {clustered_points[i]}")

    return clustered_points


def plot_clustered(clustered):
    plt.scatter([x[1][0] for x in clustered], [y[1][1] for y in clustered], c = [z[0] for z in clustered])
    plt.show()


if __name__ == "__main__":
    file_path = sys.argv[1]
    data = load_data(file_path)
    data = normalize_data(data)
    bins, nr_of_bins = split_into_bins(data)

    for column_bins in bins:
        mark_bins(column_bins, nr_of_bins)

    for column_bins in bins:
        mark_merge_bins(column_bins)

    new_bins = merge_all_bins(bins)

    poisson_threshold = 1e-4
    tree = construct_candidate_tree_start(data, new_bins)
    ns = construct_new_level(0, tree, poisson_threshold)
    candidate_list = get_candidates(ns)
    inv_cov_cluster_dict = get_inv_cov_cluster_dict(candidate_list)
    result, gmm, means = get_result(data, candidate_list, inv_cov_cluster_dict)
    means_after_bgm, cluster_dict, cluster_points = get_clusters_and_means(candidate_list, data, result, means)
    plot_means(data, means, means_after_bgm, result)
    clustered = find_outliers(data, candidate_list, cluster_dict, cluster_points, means_after_bgm, result)
    
    