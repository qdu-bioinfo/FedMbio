import torch
import torch.nn as nn
import numpy as np
from ete3 import Tree
def transform_image_metadr(X, zigzag=False):
    X = np.array(X)
    if X.size == 0:
        return torch.zeros((1, 1, 10, 10))
    raw_dim = X.shape[1]
    img_size = int(np.ceil(raw_dim ** 0.5))
    new_dim = img_size ** 2
    pad_len = new_dim - raw_dim
    pad = np.zeros((X.shape[0], pad_len))
    new_X = np.hstack((X, pad)).reshape(X.shape[0], img_size, img_size)
    if zigzag:
        for img in new_X:
            for row in range(img.shape[0]):
                if row % 2 != 0:
                    img[row] = img[row][::-1]
    flat = new_X.flatten()
    if np.all(flat == 0):
        return torch.tensor(new_X[:, np.newaxis, :, :], dtype=torch.float32)
    quantiles = np.quantile(flat, np.linspace(0, 1, 11))
    bins = [[quantiles[i], quantiles[i + 1]] for i in range(10)]
    color_vals = [0.1 * (i + 1) for i in range(10)]
    colored_X = new_X.copy()
    for i, (low, high) in enumerate(bins):
        if i == 9:
            mask = (new_X >= low) & (new_X <= high)
        else:
            mask = (new_X >= low) & (new_X < high)
        colored_X[mask] = color_vals[i]
    return torch.tensor(colored_X[:, np.newaxis, :, :], dtype=torch.float32)

def get_metadr_feature_order(tree_path, current_features):
    try:
        tree = Tree(tree_path, format=1)
    except Exception as e:
        print(f"[MetaDR Warning] Could not read tree: {tree_path}. Using raw order. Error: {e}")
        return list(current_features), list(current_features)

    level_order = [leaf.name for leaf in tree.traverse("levelorder") if leaf.is_leaf()]
    post_order = [leaf.name for leaf in tree.traverse("postorder") if leaf.is_leaf()]
    taxa_level = [i for i in level_order if i in current_features]
    taxa_post = [i for i in post_order if i in current_features]
    if len(taxa_level) == 0:
        return list(current_features), list(current_features)
    return taxa_level, taxa_post

class MetaDR_CNN(nn.Module):
    def __init__(self, num_classes, input_shape):
        super(MetaDR_CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1),
            nn.Conv2d(20, 50, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        )
        dummy_input = torch.zeros(1, 1, input_shape[0], input_shape[1])
        with torch.no_grad():
            dummy_output = self.features(dummy_input)
        flatten_dim = dummy_output.view(1, -1).size(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flatten_dim, 500),
            nn.ReLU(),
            nn.Linear(500, num_classes)
        )
    def forward(self, x):
        x = self.features(x)
        return self.fc(x)