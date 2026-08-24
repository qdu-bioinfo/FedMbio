import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from FL.clients.clientbase import load_item
from CRC.Main_Methods import load_CRC_data
from torch.utils.data import DataLoader
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_type', type=str, help='Select data type: WGS or 16S')
parser.add_argument("--groups", type=str, choices=["CTR_CRC", "CTR_ADA"],
                    help="Class pair used for the experiment: CTR_CRC or CTR_ADA")
parser.add_argument('--num_clients', type=int)
parser.add_argument('--model_dir', type=str,)
args = parser.parse_args()
num_folds = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loss_mse = torch.nn.MSELoss()
args.groups = args.groups.replace('_', '')
def load_test_data(client_id, times, groups, batch_size=16):
    test_data = load_CRC_data(
        client_id,
        f'datasetDir/{args.data_type}',
        k_fold=times,
        groups=groups,
        is_train=False
    )
    return (
        DataLoader(test_data, batch_size=batch_size, drop_last=False, shuffle=False),
        test_data[0][0].shape[0]
    )

def test_metrics(times, client_id, model_state_path, model_dir, groups):
    testloader, data_dim = load_test_data(
        client_id,
        times,
        groups=groups
    )
    model = load_item(f'fold{times}_Client{client_id}', 'model', model_state_path)
    model.eval()
    global_protos = load_item(
        f'fold{times}_Server',
        'global_protos',
        model_dir
    )
    correct_num = 0
    test_num = 0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for (x_gene, x_meta, y) in testloader:
            x_gene = x_gene.to(DEVICE)
            y = y.to(DEVICE)
            rep = model(x_gene, x_meta)
            dists = float('inf') * torch.ones(
                y.size(0), len(global_protos)
            ).to(DEVICE)
            for i, r in enumerate(rep):
                for j, pro in global_protos.items():
                    dists[i, j] = loss_mse(r, pro.to(DEVICE))
            output = model.classifier(rep)
            prob_local = torch.sigmoid(output)
            prob_local = torch.cat([1 - prob_local, prob_local], dim=1)
            probs_global = torch.softmax(-dists, dim=1)
            local_conf, _ = torch.max(prob_local, dim=1, keepdim=True)
            alpha = (local_conf - 0.5) * 2
            probs = (1 - alpha) * probs_global + alpha * prob_local
            correct_num += torch.sum(torch.argmax(probs, dim=1) == y).item()
            test_num += y.size(0)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    acc = correct_num / test_num
    auc = roc_auc_score(all_labels, all_probs[:, 1])
    return acc, auc

if __name__ == "__main__":
    fold_avg_accs = []
    fold_avg_aucs = []
    for fold in range(num_folds):
        print(f"\n========== Evaluating Fold {fold} ==========")
        fold_client_accs = []
        fold_client_aucs = []
        for client_id in range(args.num_clients):
            acc, auc = test_metrics(
                times=fold,
                client_id=client_id,
                model_state_path=args.model_dir,
                model_dir=args.model_dir,
                groups=args.groups
            )
            fold_client_accs.append(acc)
            fold_client_aucs.append(auc)
            print(
                f"[Fold {fold} | Client {client_id}] "
                f"Acc: {acc:.4f}, AUC: {auc:.4f}"
            )
        fold_mean_acc = np.mean(fold_client_accs)
        fold_mean_auc = np.mean(fold_client_aucs)
        fold_avg_accs.append(fold_mean_acc)
        fold_avg_aucs.append(fold_mean_auc)
        print(
            f"[Fold {fold} Average] "
            f"Acc: {fold_mean_acc:.4f}, "
            f"AUC: {fold_mean_auc:.4f}"
        )
    final_acc = np.mean(fold_avg_accs)
    final_auc = np.mean(fold_avg_aucs)
    print("\n========== Final 5-Fold Result ==========")
    print(f"Average Acc: {final_acc:.4f}")
    print(f"Average AUC: {final_auc:.4f}")