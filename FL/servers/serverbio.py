import torch
import torch.nn as nn
import torch.nn.functional as F
from FL.clients.clientbio import clientBio
from FL.servers.serverbase import Server
from FL.clients.clientbase import load_item, save_item
from collections import defaultdict
from torch.utils.data import DataLoader
from tqdm import trange

class FedMbio(Server):
    def __init__(self, args, tiems):
        super().__init__(args, tiems)
        self.set_clients(clientBio)
        self.times_auc = 0.0
        self.num_classes = args.num_classes
        self.server_learning_rate = args.local_learning_rate
        self.batch_size = args.batch_size
        self.server_epochs = args.server_epochs
        self.feature_dim = args.feature_dim
        self.server_hidden_dim = self.feature_dim
        if args.save_folder_name == 'temp' or 'temp' not in args.save_folder_name:
            Bio = Trainable_Global_Prototypes(
                self.num_classes,
                self.server_hidden_dim,
                self.feature_dim,
                self.device
            ).to(self.device)
            save_item(Bio, self.role, 'Bio', self.save_folder_name)
        self.CEloss = nn.CrossEntropyLoss()
        self.MSEloss = nn.MSELoss()

    def train(self, times):
        self.times_auc = 0.0
        for _ in trange(self.global_rounds + 1, desc=f"Training Fold {times}", leave=True):
            self.selected_clients = self.select_clients()
            self.evaluate(times)
            for client in self.selected_clients:
                client.train(times=times)
            self.receive_protos()
            self.update_Bio()
        round_auc_idx = self.rs_test_auc.index(max(self.rs_test_auc))
        round_auc = self.rs_test_auc[round_auc_idx]
        corresponding_acc_at_round_auc = self.rs_test_acc[round_auc_idx]
        print(
            f"The results for the {times}st fold: auc={round_auc:.3f}, acc={corresponding_acc_at_round_auc:.3f}")
        self.save_results(times=times)

    def receive_protos(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_protos = []
        uploaded_protos_per_client = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            protos = load_item(client.role, 'protos', client.save_folder_name)
            for k in protos.keys():
                if isinstance(protos[k], list):
                    for p in protos[k]:
                        self.uploaded_protos.append((p, k))
                else:
                    self.uploaded_protos.append((protos[k], k))
            uploaded_protos_per_client.append(protos)

    def update_Bio(self):
        Bio = load_item(self.role, 'Bio', self.save_folder_name)
        Bio_opt = torch.optim.SGD(Bio.parameters(), lr=self.server_learning_rate)
        Bio.train()
        for e in range(self.server_epochs):
            proto_loader = DataLoader(self.uploaded_protos, self.batch_size,
                                      drop_last=False, shuffle=True)
            for proto, y in proto_loader:
                y = torch.Tensor(y).type(torch.int64).to(self.device)
                proto_gen = Bio(list(range(self.num_classes)))
                features_square = torch.sum(torch.pow(proto, 2), 1, keepdim=True)
                centers_square = torch.sum(torch.pow(proto_gen, 2), 1, keepdim=True)
                features_into_centers = torch.matmul(proto, proto_gen.T)
                dist = features_square - 2 * features_into_centers + centers_square.T
                dist = torch.sqrt(dist)
                one_hot = F.one_hot(y, self.num_classes).to(self.device)
                dist = dist + one_hot
                loss = self.CEloss(-dist, y)
                Bio_opt.zero_grad()
                loss.backward()
                Bio_opt.step()
        self.uploaded_protos = []
        save_item(Bio, self.role, 'Bio', self.save_folder_name)
        Bio.eval()
        global_protos = defaultdict(list)
        for class_id in range(self.num_classes):
            global_protos[class_id] = Bio(torch.tensor(class_id, device=self.device)).detach()
        save_item(global_protos, self.role, 'global_protos', self.save_folder_name)

def proto_cluster(protos_list):
    proto_clusters = defaultdict(list)
    for protos in protos_list:
        for k in protos.keys():
            proto_clusters[k].append(protos[k])
    for k in proto_clusters.keys():
        protos = torch.stack(proto_clusters[k])
        proto_clusters[k] = torch.mean(protos, dim=0).detach()
    return proto_clusters


class Trainable_Global_Prototypes(nn.Module):
    def __init__(self, num_classes, server_hidden_dim, feature_dim, device):
        super().__init__()
        self.device = device
        self.embedings = nn.Embedding(num_classes, feature_dim)
        layers = [nn.Sequential(
            nn.Linear(feature_dim, server_hidden_dim),
            nn.ReLU()
        )]
        self.middle = nn.Sequential(*layers)
        self.fc = nn.Linear(server_hidden_dim, feature_dim)

    def forward(self, class_id):
        class_id = torch.tensor(class_id, device=self.device)
        emb = self.embedings(class_id)
        mid = self.middle(emb)
        out = self.fc(mid)
        return out