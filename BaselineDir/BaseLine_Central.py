import os
import pickle
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from BaselineDir.Models.MetaDR import (
    MetaDR_CNN,
    transform_image_metadr,
    get_metadr_feature_order
)
from BaselineDir.Models.PMCNN import PMCNN_Net
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

parser = argparse.ArgumentParser()

parser.add_argument('--data_type', type=str, help='Select data type: wgs or 16s')
parser.add_argument("--groups", type=str, choices=["CTR_ADA", "CTR_CRC"],
                    help="Class pair used for the experiment: CTR_CRC or CTR_ADA")
args = parser.parse_args()
args.groups = args.groups.replace('_', '')
data_dir = f"../datasetDir_Central/{args.data_type}"
tree_path = f"Distance/NWK/{args.data_type}/phylogeny.nwk"
pmcnn_list_path = rf"Distance/{args.data_type}/{''.join(args.groups)}/PMCNN_list_central.csv"
num_folds = 5
BATCH_SIZE = 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_pmcnn_grouping(nwk_path, list_path, featuresName):
    df = pd.read_csv(list_path, header=None)
    row_list = []
    for index, row in df.iterrows():
        ordered_feats = row.dropna().values
        valid_feats = [str(f) for f in ordered_feats if str(f) in featuresName]
        if len(valid_feats) > 0:
            row_list.append(valid_feats)
        else:
            row_list.append(list(featuresName[:10]))
    while len(row_list) < 4:
        row_list.append(list(featuresName))
    return row_list

class PMCNNDataset(Dataset):
    def __init__(self, X_df, y, grouping_list, min_len=8):
        self.y = torch.tensor(y, dtype=torch.long)
        self.data_branches = []
        for feats in grouping_list:
            valid = [f for f in feats if f in X_df.columns]
            if len(valid) == 0:
                branch_data = np.zeros((len(X_df), min_len), dtype=np.float32)
            else:
                data = X_df[valid].values.astype(np.float32)
                if data.shape[1] < min_len:
                    pad_width = min_len - data.shape[1]
                    data = np.pad(
                        data,
                        ((0, 0), (0, pad_width)),
                        mode="constant"
                    )
                branch_data = data
            self.data_branches.append(torch.tensor(branch_data))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        inputs = [b[idx] for b in self.data_branches]
        return tuple(inputs), self.y[idx]

def train_evaluate_dl(X_train, y_train, X_test, model_name, featuresName):
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    if model_name == "MetaDR":
        lvl, post = get_metadr_feature_order(tree_path, featuresName)
        Xtr = pd.DataFrame(X_train, columns=featuresName)
        Xte = pd.DataFrame(X_test, columns=featuresName)
        Xtr_l = transform_image_metadr(
            Xtr[lvl].values, zigzag=False
        ).to(device)
        Xtr_p = transform_image_metadr(
            Xtr[post].values, zigzag=True
        ).to(device)
        Xte_l = transform_image_metadr(
            Xte[lvl].values, zigzag=False
        ).to(device)
        Xte_p = transform_image_metadr(
            Xte[post].values, zigzag=True
        ).to(device)
        img_shape = (Xtr_l.shape[2], Xtr_l.shape[3])
        model_l = MetaDR_CNN(2, img_shape).to(device)
        model_p = MetaDR_CNN(2, img_shape).to(device)
        ds_l = TensorDataset(Xtr_l, y_train_t)
        loader_l = DataLoader(ds_l, batch_size=BATCH_SIZE, shuffle=True)
        opt_l = optim.Adam(model_l.parameters(), lr=1e-3)
        model_l.train()
        for _ in range(20):
            for xb, yb in loader_l:
                opt_l.zero_grad()
                loss = nn.CrossEntropyLoss()(model_l(xb), yb)
                loss.backward()
                opt_l.step()
        ds_p = TensorDataset(Xtr_p, y_train_t)
        loader_p = DataLoader(ds_p, batch_size=BATCH_SIZE, shuffle=True)
        opt_p = optim.Adam(model_p.parameters(), lr=1e-3)
        model_p.train()
        for _ in range(20):
            for xb, yb in loader_p:
                opt_p.zero_grad()
                loss = nn.CrossEntropyLoss()(model_p(xb), yb)
                loss.backward()
                opt_p.step()
        model_l.eval()
        model_p.eval()
        with torch.no_grad():
            p1 = torch.softmax(model_l(Xte_l), dim=1)[:, 1]
            p2 = torch.softmax(model_p(Xte_p), dim=1)[:, 1]
        return ((p1 + p2) / 2).cpu().numpy()
    if model_name == "PMCNN":
        grouping = get_pmcnn_grouping(tree_path, pmcnn_list_path, featuresName)
        train_ds = PMCNNDataset(
            pd.DataFrame(X_train, columns=featuresName),
            y_train_t.cpu(),
            grouping
        )
        test_ds = PMCNNDataset(
            pd.DataFrame(X_test, columns=featuresName),
            torch.zeros(len(X_test)),
            grouping
        )
        branch_lens = [b.shape[1] for b in train_ds.data_branches]
        model = PMCNN_Net(branch_lens).to(device)
        opt = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
        model.train()
        for _ in range(20):
            for inputs, labels in loader:
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                opt.zero_grad()
                nn.CrossEntropyLoss()(model(*inputs), labels).backward()
                opt.step()
        model.eval()
        probs = []
        with torch.no_grad():
            for inputs, _ in DataLoader(test_ds, batch_size=BATCH_SIZE):
                inputs = [x.to(device) for x in inputs]
                probs.append(torch.softmax(model(*inputs), 1)[:, 1])
        return torch.cat(probs).cpu().numpy()
    if model_name == "CNN":
        Xtr = torch.tensor(X_train, dtype=torch.float32).unsqueeze(1).to(device)
        Xte = torch.tensor(X_test, dtype=torch.float32).unsqueeze(1).to(device)
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, 2)
                )
            def forward(self, x):
                return self.net(x)
        model = SimpleCNN().to(device)
        opt = optim.Adam(model.parameters(), lr=0.01)
        loader = DataLoader(TensorDataset(Xtr, y_train_t),
                            batch_size=BATCH_SIZE, shuffle=True)
        model.train()
        for _ in range(50):
            for xb, yb in loader:
                opt.zero_grad()
                nn.CrossEntropyLoss()(model(xb), yb).backward()
                opt.step()
        with torch.no_grad():
            return torch.softmax(model(Xte), 1)[:, 1].cpu().numpy()
class MLPNetwork(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 2),
        )
    def forward(self, X):
        return self.network(X)
class MLPClassifier:
    def __init__(self):
        self.learning_rate = 0.001
        self.epochs = 200
        self.batch_size = 16
        self.model = None
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).reshape(-1)
        self.classes_ = np.unique(y)
        if self.classes_.size != 2:
            raise ValueError("MLPClassifierSGD supports binary classification only.")
        y_binary = (y == self.classes_[1]).astype(np.int64)
        X_tensor = torch.from_numpy(X)
        y_tensor = torch.from_numpy(y_binary)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.input_size = X.shape[1]
        self.model = MLPNetwork(self.input_size).to(self.device)
        generator = torch.Generator()
        train_loader = DataLoader(
            TensorDataset(X_tensor, y_tensor),
            batch_size=min(self.batch_size, X.shape[0]),
            shuffle=True,
            generator=generator,
        )
        optimizer = optim.SGD(self.model.parameters(), lr=self.learning_rate)
        criterion = nn.CrossEntropyLoss()
        self.model.train()
        for _ in range(self.epochs):
            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                optimizer.zero_grad()
                loss = criterion(self.model(X_batch), y_batch)
                loss.backward()
                optimizer.step()
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("Call fit before predict_proba.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.shape[1] != self.input_size:
            raise ValueError("X has an incompatible feature dimension.")
        if not np.isfinite(X).all():
            raise ValueError("X contains NaN or infinite values.")
        self.model.eval()
        with torch.no_grad():
            logits = self.model(torch.from_numpy(X).to(self.device))
            return torch.softmax(logits, dim=1).cpu().numpy()

def build_model(name):
    if name == "rf":
        return RandomForestClassifier(random_state=42)
    if name == "knn":
        return KNeighborsClassifier(5)
    if name == "mlp":
        return MLPClassifier()
    raise ValueError(name)

def run_one_model(model_name):
    accs, aucs = [], []
    for fold in range(num_folds):
        with open(os.path.join(data_dir, f"{''.join(args.groups)}_fold{fold}.pkl"), "rb") as f:
            data = pickle.load(f)
        X_train, y_train = data["X_mic_train"], data["y_train"]
        X_test, y_test = data["X_mic_test"], data["y_test"]
        new_features = []
        for col in data["featuresName"]:
            col = str(col)
            if col.startswith("OTU"):
                new_features.append(col.replace("-", "."))
            else:
                new_features.append(f"s__{col}")
        data["featuresName"] = new_features
        featuresName = data["featuresName"]
        if model_name in ["CNN", "PMCNN", "MetaDR"]:
            y_prob = train_evaluate_dl(X_train, y_train, X_test,
                                       model_name, featuresName)
        else:
            model = build_model(model_name)
            model.fit(X_train, y_train)
            y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        print(f"Fold {fold} | {model_name} | ACC={acc:.4f} AUC={auc:.4f}")
        accs.append(acc)
        aucs.append(auc)
    print(f"{model_name} OVERALL ACC={np.mean(accs):.4f} AUC={np.mean(aucs):.4f}")
    return np.mean(accs), np.mean(aucs)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(42)
    models = ["rf", "mlp", "knn", "PMCNN", "MetaDR", "CNN", ]
    results = {}
    for m in models:
        acc, auc = run_one_model(m)
        results[m] = (acc, auc)
    print("\n===== FINAL SUMMARY =====")
    for m, (acc, auc) in results.items():
        print(f"{m:<8} | ACC={acc:.4f} | AUC={auc:.4f}")