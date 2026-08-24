import os
import pickle
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
from Models.MetaDR import MetaDR_CNN, transform_image_metadr, get_metadr_feature_order
from Models.PMCNN import PMCNN_Net

def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
def get_pmcnn_grouping(nwk_path, list_path, feature_names):
    df = pd.read_csv(list_path, header=None)
    row_list = []
    for index, row in df.iterrows():
        ordered_feats = row.dropna().values
        valid_feats = [str(f) for f in ordered_feats if str(f) in feature_names]
        if len(valid_feats) > 0:
            row_list.append(valid_feats)
        else:
            row_list.append(list(feature_names[:10]))
    while len(row_list) < 4:
        row_list.append(list(feature_names))
    return row_list

class PMCNNDataset(Dataset):
    def __init__(self, X_df, y, grouping_list):
        self.y = y
        self.data_branches = []
        for feats in grouping_list:
            if len(feats) == 0:
                branch_data = np.zeros((len(X_df), 10), dtype=np.float32)
            else:
                valid = [f for f in feats if f in X_df.columns]
                if len(valid) == 0:
                    branch_data = np.zeros((len(X_df), 10), dtype=np.float32)
                else:
                    branch_data = X_df[valid].values.astype(np.float32)
            self.data_branches.append(torch.tensor(branch_data))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        inputs = [branch[index] for branch in self.data_branches]
        return tuple(inputs), self.y[index]

def train_evaluate_dl(X_train_raw, y_train, X_test_raw, model_type, feature_names, batch_size=32):
    y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
    y_test_dummy = torch.zeros(len(X_test_raw), dtype=torch.long)
    if model_type == 'MetaDR':
        level_names, post_names = get_metadr_feature_order(tree_path, feature_names)
        X_train_df = pd.DataFrame(X_train_raw, columns=feature_names)
        X_test_df = pd.DataFrame(X_test_raw, columns=feature_names)
        X_train_l = transform_image_metadr(X_train_df[level_names].values, zigzag=False).to(device)
        X_train_p = transform_image_metadr(X_train_df[post_names].values, zigzag=True).to(device)
        X_test_l = transform_image_metadr(X_test_df[level_names].values, zigzag=False).to(device)
        X_test_p = transform_image_metadr(X_test_df[post_names].values, zigzag=True).to(device)
        img_shape = (X_train_l.shape[2], X_train_l.shape[3])
        model_l = MetaDR_CNN(2, img_shape).to(device)
        model_p = MetaDR_CNN(2, img_shape).to(device)
        ds_l = TensorDataset(X_train_l, y_train_t)
        ds_p = TensorDataset(X_train_p, y_train_t)
        opt_l = optim.Adam(model_l.parameters(), lr=1e-3)
        model_l.train()
        for epoch in range(20):
            for xb, yb in DataLoader(ds_l, batch_size=batch_size, shuffle=True):
                opt_l.zero_grad()
                loss = nn.CrossEntropyLoss()(model_l(xb), yb)
                loss.backward()
                opt_l.step()
        opt_p = optim.Adam(model_p.parameters(), lr=1e-3)
        model_p.train()
        for epoch in range(20):
            for xb, yb in DataLoader(ds_p, batch_size=batch_size, shuffle=True):
                opt_p.zero_grad()
                loss = nn.CrossEntropyLoss()(model_p(xb), yb)
                loss.backward()
                opt_p.step()
        model_l.eval()
        model_p.eval()
        with torch.no_grad():
            p1 = torch.softmax(model_l(X_test_l), 1).cpu().numpy()
            p2 = torch.softmax(model_p(X_test_p), 1).cpu().numpy()
        return (p1 + p2)[:, 1] / 2.0
    elif model_type == 'PMCNN':
        grouping_list = get_pmcnn_grouping(tree_path, pmcnn_list_path, feature_names)
        train_ds = PMCNNDataset(pd.DataFrame(X_train_raw, columns=feature_names), y_train_t.cpu(), grouping_list)
        test_ds = PMCNNDataset(pd.DataFrame(X_test_raw, columns=feature_names), y_test_dummy, grouping_list)
        branch_lens = [t.shape[1] for t in train_ds.data_branches]
        model = PMCNN_Net(branch_lens).to(device)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.0001)
        model.train()
        for epoch in range(20):
            for inputs, labels in train_loader:
                inputs = [x.to(device) for x in inputs]
                labels = labels.to(device)
                optimizer.zero_grad()
                out = model(*inputs)
                loss = nn.CrossEntropyLoss()(out, labels)
                loss.backward()
                optimizer.step()
        model.eval()
        probs = []
        test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            for inputs, _ in test_loader:
                inputs = [x.to(device) for x in inputs]
                out = model(*inputs)
                probs.extend(torch.softmax(out, 1)[:, 1].cpu().numpy())
        return np.array(probs)
    elif model_type == 'CNN':
        X_train_t = torch.tensor(X_train_raw, dtype=torch.float32).unsqueeze(1).to(device)
        X_test_t = torch.tensor(X_test_raw, dtype=torch.float32).unsqueeze(1).to(device)
        class QuickCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv1d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(),
                    nn.Linear(32, 2)
                )
            def forward(self, x): return self.net(x)
        model = QuickCNN().to(device)
        train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True)
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        model.train()
        for epoch in range(50):
            for xb, yb in train_loader:
                optimizer.zero_grad()
                nn.CrossEntropyLoss()(model(xb), yb).backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():
            out = model(X_test_t)
            probs = torch.softmax(out, 1)[:, 1].cpu().numpy()
        return probs
    return np.zeros(len(X_test_raw))
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

def build_sklearn_model(model_name):
    if model_name == "rf":
        return RandomForestClassifier(random_state=42)
    elif model_name == "knn":
        return KNeighborsClassifier(n_neighbors=5)
    elif model_name == "mlp":
        return MLPClassifier()
    return None

def train_evaluate(X_train, y_train, X_test, y_test, client_id, fold, model_name, feature_names, batch_size=32):
    if model_name in ['CNN', 'PMCNN', 'MetaDR']:
        y_prob = train_evaluate_dl(X_train, y_train, X_test, model_name, feature_names, batch_size=batch_size)
    else:
        model = build_sklearn_model(model_name)
        model.fit(X_train, y_train)
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)
    y_pred = (y_prob >= 0.5).astype(int)
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    print(f'Fold {fold} Client {client_id} ({model_name}) - ACC: {acc:.4f}, AUC: {auc:.4f}')
    return acc, auc

def run_one_model(model_name, batch_size=32):
    fold_acc, fold_auc = [], []
    print(f"\n===== Running {model_name} =====")
    for fold in range(num_folds):
        accs, aucs = [], []
        for client_id in range(args.num_clients):
            file_path = os.path.join(data_dir, f'{"".join(args.groups)}_fold{fold}_study{client_id}.pkl')
            if not os.path.exists(file_path):
                continue
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            X_train = data['X_mic_train']
            y_train = data['y_train']
            X_test = data['X_mic_test']
            y_test = data['y_test']
            feat_names = data.get('featuresName')
            if feat_names is None:
                feat_names = [str(i) for i in range(X_train.shape[1])]
            else:
                feat_names = [str(x) for x in feat_names]
            acc, auc = train_evaluate(
                X_train, y_train, X_test, y_test,
                client_id, fold, model_name, feat_names, batch_size=batch_size
            )
            accs.append(acc)
            aucs.append(auc)
        if len(accs) > 0:
            print(f"Fold {fold} AVG: ACC={np.mean(accs):.4f}, AUC={np.mean(aucs):.4f}")
            fold_acc.append(np.mean(accs))
            fold_auc.append(np.mean(aucs))
    overall_acc = np.mean(fold_acc) if fold_acc else 0
    overall_auc = np.mean(fold_auc) if fold_auc else 0
    print(f"{model_name} Overall: ACC={overall_acc:.4f}, AUC={overall_auc:.4f}")
    return overall_acc, overall_auc


if __name__ == "__main__":
    set_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, help='Select data type: wgs or 16s')
    parser.add_argument("--groups", type=str, choices=["CTR_ADA", "CTR_CRC"],
                        help="Class pair used for the experiment: CTR_CRC or CTR_ADA")
    parser.add_argument('-nc', "--num_clients", type=int, help="Total number of institutions/clients")
    args = parser.parse_args()
    args.groups = args.groups.replace('_', '')
    num_folds = 5
    BATCH_SIZE = 16
    tree_path = f"Distance/NWK/{args.data_type}/phylogeny.nwk"
    pmcnn_list_path = rf"Distance/{args.data_type}/{''.join(args.groups)}/PMCNN_list.csv"
    data_dir = f'../datasetDir/{args.data_type}'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    models = ["rf", "mlp", "knn", "PMCNN", "MetaDR", "CNN", ]
    results = {}
    for m in models:
        acc, auc = run_one_model(m, batch_size=BATCH_SIZE)
        results[m] = (acc, auc)
    print("\n===== Final Summary =====")
    for m, (acc, auc) in results.items():
        print(f"{m:<10} | ACC: {acc:.4f} | AUC: {auc:.4f}")
