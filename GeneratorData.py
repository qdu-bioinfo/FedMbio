import os
os.environ["SCIPY_ARRAY_API"] = "1"
import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import random
import torch
import argparse
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from FL.utils import HCFR


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


def generate_CRC_dataset_kfold(groups, data_path, meta_path, output_dir, distance_path, n_splits=5, random_seed=42):
    os.makedirs(output_dir, exist_ok=True)
    meta = pd.read_csv(meta_path, sep=",", index_col=0)
    meta = meta[meta.Group.isin(groups)]
    meta = meta[['Group', 'Country', 'Project', 'Study', 'Age', 'BMI', 'Sex']].fillna('NULL')
    group_to_num = {group: i for i, group in enumerate(groups)}
    meta_samples = set(meta.index)
    client_no = -1
    for filename in os.listdir(data_path):
        if filename.endswith('.csv') and filename != 'meta.csv':
            file_path = os.path.join(data_path, filename)
            features = pd.read_csv(file_path, index_col=0)
            common_samples = features.index.intersection(meta_samples)
            features_client = features.loc[common_samples]
            meta_client = meta.loc[common_samples]
            if len(meta_client['Group'].unique()) < 2:
                continue
            else:
                client_no += 1
            y = np.array([group_to_num[label] for label in meta_client["Group"].values])
            meta_client = meta_client[['Country', 'Project', 'Age', 'BMI', 'Sex']]
            features_client = preprocess_microbiome(features_client, clr=True, min_prev=0.25)
            new_columns = []
            for col in features_client.columns:
                if col.startswith("OTU"):
                    new_columns.append(col.replace('-', '.'))
                else:
                    new_columns.append(f"s__{col}")
            features_client.columns = new_columns
            full_feature_order = HCFR.hac(distance_path)
            feature_order = [f for f in full_feature_order if f in features_client.columns]
            features_client = features_client[feature_order]
            skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_seed)
            for fold, (train_idx, test_idx) in enumerate(skf.split(features_client, y)):
                featuresName = features_client.columns.tolist()
                train_samplesName = features_client.iloc[train_idx].index.tolist()
                test_samplesName = features_client.iloc[test_idx].index.tolist()
                X_mic_train = features_client.iloc[train_idx]
                X_mic_test = features_client.iloc[test_idx]
                X_meta_train = meta_client.iloc[train_idx]
                X_meta_test = meta_client.iloc[test_idx]
                y_train = y[train_idx]
                y_test = y[test_idx]
                train_samplesName_arr = np.array(train_samplesName)
                if len(common_samples) > 200:
                    rus = RandomUnderSampler(random_state=random_seed)
                    X_mic_train, y_train = rus.fit_resample(X_mic_train, y_train)
                    resampled_idx = rus.sample_indices_
                    train_samplesName = train_samplesName_arr[resampled_idx].tolist()
                    X_meta_train = X_meta_train.iloc[resampled_idx]
                else:
                    smote = SMOTE(random_state=random_seed)
                    X_mic_train_resampled, y_train_resampled = smote.fit_resample(X_mic_train, y_train)
                    n_original = X_mic_train.shape[0]
                    n_resampled = X_mic_train_resampled.shape[0]
                    n_synthetic = n_resampled - n_original
                    synthetic_names = [f"Synthetic_{i}" for i in range(n_synthetic)]
                    train_samplesName = train_samplesName + synthetic_names
                    synthetic_labels = y_train_resampled[n_original:]
                    synthetic_meta_rows = []
                    for label in synthetic_labels:
                        indices = np.where(y_train == label)[0]
                        chosen_idx = np.random.choice(indices)
                        synthetic_meta_rows.append(X_meta_train.iloc[chosen_idx])
                    if len(synthetic_meta_rows) > 0:
                        synthetic_meta_df = pd.DataFrame(synthetic_meta_rows)
                        synthetic_meta_df.index = range(n_original, n_resampled)
                        X_meta_train = pd.concat([X_meta_train, synthetic_meta_df], ignore_index=True)
                    X_mic_train = X_mic_train_resampled
                    y_train = y_train_resampled
                scaler = StandardScaler()
                X_mic_train = scaler.fit_transform(X_mic_train)
                X_mic_test = scaler.transform(X_mic_test)

                def meta_row_to_text(row):
                    return f"This sample from {row['Country']}, Project {row['Project']}, Age {row['Age']}, BMI{row['BMI']}, Sex {row['Sex']}"

                X_meta_train = X_meta_train.apply(meta_row_to_text, axis=1).tolist()
                X_meta_test = X_meta_test.apply(meta_row_to_text, axis=1).tolist()
                fold_data = {
                    'featuresName': featuresName,
                    'train_samplesName': train_samplesName,
                    'X_mic_train': X_mic_train,
                    'X_meta_train': X_meta_train,
                    'test_samplesName': test_samplesName,
                    'X_mic_test': X_mic_test,
                    'X_meta_test': X_meta_test,
                    'y_train': y_train,
                    'y_test': y_test,
                }
                fold_file = os.path.join(output_dir, f'{"".join(groups)}_fold{fold}_study{client_no}.pkl')
                with open(fold_file, 'wb') as f:
                    pickle.dump(fold_data, f)
                print(f"Saved fold {fold} for study {client_no} ({filename}) Mic_dim={X_mic_train.shape[1]}")
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
    distance_path = f'BaselineDir/Distance/distance_matrix/{args.data_type}/distance_matrix.csv'
    output_dir = f'datasetDir/{args.data_type}'
    generate_CRC_dataset_kfold(groups, data_path,
                               meta_path, output_dir, distance_path, n_splits=5, random_seed=42)
