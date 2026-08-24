import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
import random
import torch
import argparse

def preprocess_microbiome(mat: pd.DataFrame,
                          clr: bool = True,
                          min_prev: float = 0.10) -> pd.DataFrame:
    arr = mat.values.astype(np.float32)
    if clr:
        gm = np.exp(np.mean(np.log(arr + 1e-9), axis=1)[:, None])
        arr = np.log(arr + 1e-9) - np.log(gm)
    prevalence = (mat.values > 1e-8).mean(axis=0)
    keep = prevalence > min_prev
    return pd.DataFrame(arr[:, keep],
                        index=mat.index,
                        columns=mat.columns[keep])
def generate_CRC_dataset_merge_cv(groups, data_path, meta_path, output_dir,
                                  n_splits=5, random_seed=42):
    random.seed(random_seed)
    np.random.seed(random_seed)
    os.makedirs(output_dir, exist_ok=True)
    meta = pd.read_csv(meta_path, sep=",", index_col=0)
    meta = meta[meta.Group.isin(groups)]
    meta = meta[['Group', 'Country', 'Project', 'Study', 'Age', 'BMI', 'Sex']].fillna('NULL')
    group_to_num = {group: i for i, group in enumerate(groups)}
    all_gene, all_meta, all_y = [], [], []

    for filename in os.listdir(data_path):
        if not filename.endswith(".csv") or filename == "meta.csv":
            continue
        file_path = os.path.join(data_path, filename)
        features = pd.read_csv(file_path, index_col=0)
        common_samples = features.index.intersection(meta.index)
        features = features.loc[common_samples]
        meta_client = meta.loc[common_samples]
        if len(meta_client['Group'].unique()) < 2:
            print(f"Skip {filename}, only one group {meta_client['Group'].unique()}")
            continue
        y = np.array([group_to_num[label] for label in meta_client["Group"].values])
        meta_client = meta_client[['Country', 'Project', 'Age', 'BMI', 'Sex']]
        features = preprocess_microbiome(features, clr=True, min_prev=0.25)
        all_gene.append(features)
        all_meta.append(meta_client)
        all_y.append(y)
        print(f"✔ Processed {filename}")
    X_gene_all = pd.concat(all_gene, axis=0).fillna(0)
    X_meta_all = pd.concat(all_meta, axis=0)
    y_all = np.concatenate(all_y)

    scaler = StandardScaler()
    X_gene_all = pd.DataFrame(scaler.fit_transform(X_gene_all),
                              index=X_gene_all.index,
                              columns=X_gene_all.columns)

    def meta_row_to_text(row):
        return f"This sample from {row['Country']}, Project {row['Project']}, Age {row['Age']}, BMI{row['BMI']}, Sex {row['Sex']}"

    X_meta_text_all = X_meta_all.apply(meta_row_to_text, axis=1).tolist()

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X_gene_all), 0):
        featuresName = X_gene_all.columns.tolist()
        train_samplesName = X_gene_all.iloc[train_idx].index.tolist()
        test_samplesName = X_gene_all.iloc[test_idx].index.tolist()
        X_mic_train = X_gene_all.iloc[train_idx].values
        X_mic_test = X_gene_all.iloc[test_idx].values
        X_meta_train = [X_meta_text_all[i] for i in train_idx]
        X_meta_test = [X_meta_text_all[i] for i in test_idx]
        y_train = y_all[train_idx]
        y_test = y_all[test_idx]
        dataset = {
            'featuresName': featuresName,
            'train_samplesName': train_samplesName,
            'test_samplesName': test_samplesName,
            'X_mic_train': X_mic_train,
            'X_meta_train': X_meta_train,
            'X_mic_test': X_mic_test,
            'X_meta_test': X_meta_test,
            'y_train': y_train,
            'y_test': y_test,
        }

        fold_file = os.path.join(output_dir, f"{''.join(groups)}_fold{fold}.pkl")
        with open(fold_file, 'wb') as f:
            pickle.dump(dataset, f)

        print(f"✅ Saved fold {fold}: file={fold_file}")


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    setup_seed(42)
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_type', type=str, help='Select data type: wgs or 16s')
    parser.add_argument("--groups", type=str, choices=["CTR_ADA", "CTR_CRC"],
                        help="Class pair used for the experiment: CTR_CRC or CTR_ADA")
    args = parser.parse_args()
    if args.groups == "CTR_CRC":
        groups = ["CTR", "CRC"]
    elif args.groups == "CTR_ADA":
        groups = ["CTR", "ADA"]
    data_path = f'CRC/{args.data_type}/'
    meta_path = f'CRC/{args.data_type}/meta.csv'
    output_dir = f'datasetDir_Central/{args.data_type}'
    n_splits = 5
    args = parser.parse_args()
    generate_CRC_dataset_merge_cv(groups, data_path,
                                  meta_path, output_dir,
                                  n_splits=n_splits, random_seed=42)
