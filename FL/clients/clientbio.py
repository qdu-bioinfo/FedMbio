import copy
import torch
import torch.nn as nn
import numpy as np
from FL.clients.clientbase import Client, load_item, save_item
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from sklearn.cluster import KMeans

class clientBio(Client):
    def __init__(self, args, id, train_samples, test_samples, input_dim, **kwargs):
        super().__init__(args, id, train_samples, test_samples, input_dim, **kwargs)
        self.loss_mse = nn.MSELoss()

    def train(self, times):
        trainloader = self.load_train_data(times=times)
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        model.to(self.device)
        model.train()
        for step in range(self.local_epochs):
            for i, (x_gene, x_meta, y) in enumerate(trainloader):
                x_gene = x_gene.to(self.device)
                y = y.to(self.device)
                rep = model(x_gene, x_meta)
                output = model.classifier(rep)
                loss = self.loss(output, y.float().unsqueeze(1))
                if global_protos is not None:
                    proto_new = copy.deepcopy(rep.detach())
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        proto_new[i, :] = global_protos[y_c].data
                    lamda = 14.0 if self.groups == ["CTR", "CRC"] else 1.0
                    loss_dis = self.loss_mse(proto_new, rep) * lamda
                    loss += loss_dis
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        self.collect_protos(times=times)
        save_item(model, self.role, 'model', self.save_folder_name)

    def collect_protos(self, times):
        trainloader = self.load_train_data(times=times)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model = load_item(self.role, 'model', self.save_folder_name)
        model.eval()
        protos = defaultdict(list)
        with torch.no_grad():
            for _, (x_gene, x_meta, y) in enumerate(trainloader):
                x_gene = x_gene.to(self.device)
                y = y.to(self.device)
                rep = model(x_gene, x_meta)
                if(global_protos is None):
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        protos[y_c].append(rep[i, :].detach().data)
                else:
                    dists = float('inf') * torch.ones(y.shape[0], len(global_protos)).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            dists[i, j] = self.loss_mse(r, pro)
                    preds = torch.argmin(dists, dim=1)
                    for i, yy in enumerate(y):
                        y_c = yy.item()
                        if preds[i].item() == y_c:
                            protos[y_c].append(rep[i, :].detach().clone())
                        else:
                            continue
        save_item(agg_func(protos), self.role, 'protos', self.save_folder_name)

    def test_metrics(self, times):
        testloader = self.load_test_data(times, groups=self.groups)
        model = load_item(self.role, 'model', self.save_folder_name)
        global_protos = load_item('Server', 'global_protos', self.save_folder_name)
        model.eval()
        correct_num = 0
        test_num = 0
        all_probs = []
        all_labels = []
        if global_protos is not None:
            with torch.no_grad():
                for (x_gene, x_meta, y) in testloader:
                    x_gene = x_gene.to(self.device)
                    y = y.to(self.device)
                    rep = model(x_gene,x_meta)
                    dists = float('inf') * torch.ones(y.shape[0], len(global_protos)).to(self.device)
                    for i, r in enumerate(rep):
                        for j, pro in global_protos.items():
                            dists[i, j] = self.loss_mse(r, pro)
                    output = model.classifier(rep)
                    prob_local = torch.sigmoid(output)
                    prob_local = torch.cat([1 - prob_local, prob_local], dim=1)
                    probs_global = torch.softmax(-dists, dim=1)
                    local_conf, _ = torch.max(prob_local, dim=1, keepdim=True)
                    alpha = (local_conf - 0.5) * 2
                    probs = (1 - alpha) * probs_global + alpha * prob_local
                    correct_num += (torch.sum(torch.argmax(probs, dim=1) == y)).item()
                    test_num += y.shape[0]
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(y.cpu().numpy())
                all_probs = np.concatenate(all_probs, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)
                if np.isnan(all_probs).any() or np.isinf(all_probs).any():
                    all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=0.5, neginf=0.5)
                auc = roc_auc_score(all_labels, all_probs[:, 1])
                acc = correct_num * 1.0 / test_num
                return acc, auc
        else:
            with torch.no_grad():
                for (x_gene, x_meta, y) in testloader:
                    x_gene = x_gene.to(self.device)
                    y = y.to(self.device)
                    rep = model(x_gene, x_meta)
                    output = model.classifier(rep)
                    prob_local = torch.sigmoid(output)
                    prob_local = torch.cat([1 - prob_local, prob_local], dim=1)
                    probs = prob_local
                    correct_num += (torch.sum(torch.argmax(probs, dim=1) == y)).item()
                    test_num += y.shape[0]                    
                    all_probs.append(probs.cpu().numpy())
                    all_labels.append(y.cpu().numpy())            
            if len(all_probs) > 0:
                all_probs = np.concatenate(all_probs, axis=0)
                all_labels = np.concatenate(all_labels, axis=0)                
                if np.isnan(all_probs).any() or np.isinf(all_probs).any():
                     all_probs = np.nan_to_num(all_probs, nan=0.5, posinf=0.5, neginf=0.5)
                auc = roc_auc_score(all_labels, all_probs[:, 1]) 
                acc = correct_num * 1.0 / test_num if test_num > 0 else 0
                return acc, auc
            else:
                return 0, 0.5
def agg_func(protos):
    for [label, proto_list] in protos.items():
        if len(proto_list) > 20:
            model = KMeans(
                n_clusters=3,
                init="k-means++",
                n_init=10,
                random_state=0
            )
            data = torch.stack(proto_list).cpu().numpy()
            model.fit(data)
            new_protos = []
            for center in model.cluster_centers_:
                new_protos.append(torch.from_numpy(center).to(proto_list[0].device))
            protos[label] = new_protos
        elif len(proto_list)>0 & len(proto_list)<=20:
            proto = 0 * proto_list[0].data
            for i in proto_list:
                proto += i.data
            protos[label] = proto / len(proto_list)
        else:            
            protos[label] = proto_list[0]
    return protos
