import pandas as pd
import numpy as np
from scipy.cluster.hierarchy import linkage, fcluster
from Bio import Phylo
import os
def generate_grouping_file(newick_path, output_csv_path):
    tree = Phylo.read(newick_path, "newick")
    terminals = tree.get_terminals()
    num_terminals = len(terminals)
    raw_feature_names = [term.name for term in terminals]
    matrix = np.zeros((num_terminals, num_terminals))
    for i in range(num_terminals):
        for j in range(num_terminals):
            matrix[i, j] = tree.distance(terminals[i], terminals[j])
    max_dist = matrix.max()
    if max_dist == 0: max_dist = 1.0
    similarity_matrix = np.exp(-(matrix * matrix) / 0.5)
    dist_for_linkage = 1.0 - similarity_matrix
    from scipy.spatial.distance import squareform
    condensed_dist = squareform(dist_for_linkage, checks=False)
    linked = linkage(condensed_dist, method='single')
    thresholds = [0.1, 0.01, 0.3, 0.2]
    grouped_features_list = []
    for threshold in thresholds:
        clusters = fcluster(linked, t=threshold, criterion='distance')
        cluster_map = {}
        for feat_idx, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_map:
                cluster_map[cluster_id] = []
            cluster_map[cluster_id].append(feat_idx)
        sorted_cluster_ids = sorted(cluster_map.keys())
        reordered_indices = []
        for cid in sorted_cluster_ids:
            reordered_indices.extend(cluster_map[cid])
        reordered_names = [raw_feature_names[i] for i in reordered_indices]
        grouped_features_list.append(reordered_names)
    df = pd.DataFrame(grouped_features_list)
    os.makedirs(os.path.dirname(output_csv_path),exist_ok=True)
    df.to_csv(output_csv_path, index=False, header=False)