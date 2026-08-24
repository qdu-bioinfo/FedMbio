import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

def hac(path_to_csv, sigma=None):
    df = pd.read_csv(path_to_csv, index_col=0)
    D = df.values
    n = D.shape[0]
    if sigma is None:
        tri_upper = D[np.triu_indices(n, k=1)]
        sigma = np.median(tri_upper)
    S = np.exp(- (D ** 2) / (2 * sigma ** 2))
    Dprime = 1.0 - S
    condensed = squareform(Dprime)
    Z = linkage(condensed, method='single')
    ivl = dendrogram(Z, no_plot=True)['ivl']
    order = list(map(int, ivl))
    raw_feature_names = df.index.tolist()
    ordered_names = [f"{raw_feature_names[i]}" for i in order]
    return ordered_names