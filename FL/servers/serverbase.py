import torch
import os
import numpy as np
import h5py
from CRC.Main_Methods import load_CRC_data
import glob
import shutil
class Server(object):
    def __init__(self, args, times):
        self.args = args
        self.device = args.device
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.algorithm = args.algorithm
        self.role = 'Server'
        self.save_folder_name = args.save_folder_name_full
        self.clients = []
        self.selected_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.rs_test_acc = []
        self.rs_test_auc = []
        self.times = times
    def set_clients(self, clientObj):
        for i in range(self.num_clients):
            train_data = load_CRC_data(i, output_dir=self.args.datasetDir, k_fold =self.times, groups=self.args.groups, is_train=True)
            test_data = load_CRC_data(i, output_dir=self.args.datasetDir, k_fold =self.times, groups=self.args.groups, is_train=False)
            client = clientObj(self.args,
                               id=i,
                               train_samples=len(train_data),
                               test_samples=len(test_data),
                               input_dim=train_data[0][0].shape[0],)
            self.clients.append(client)
    def select_clients(self):
        self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))
        return selected_clients

    def save_global_model(self):
        model_path = os.path.join("models", self.dataset)
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        torch.save(self.global_model, model_path)

    def load_model(self):
        model_path = os.path.join("models", self.dataset)
        model_path = os.path.join(model_path, self.algorithm + "_server" + ".pt")
        assert (os.path.exists(model_path))
        self.global_model = torch.load(model_path)

    def save_results(self,times):
        if (len(self.rs_test_acc)):
            file_path = self.save_folder_name + f"/Result_{times}.h5"

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)

        pt_files = glob.glob(os.path.join(self.save_folder_name, '*.pt'))
        for file_path in pt_files:
            os.remove(file_path)

    def test_metrics(self,times):
        all_acc = []
        all_auc = []
        for c in self.clients:
            acc, auc = c.test_metrics(times=times)
            all_acc.append(acc)
            all_auc.append(auc)
        ids = [c.id for c in self.clients]
        return ids, all_acc, all_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl * 1.0)
        ids = [c.id for c in self.clients]
        return ids, num_samples, losses

    def evaluate(self, times):
        stats = self.test_metrics(times)
        acc = np.mean(stats[1])
        auc = np.mean(stats[2])
        self.rs_test_acc.append(acc)
        self.rs_test_auc.append(auc)
        if(self.times_auc+1e-9 < auc):
            self.times_auc = auc
            best_dir = os.path.join(self.save_folder_name, "result")
            os.makedirs(best_dir, exist_ok=True)
            pt_files = glob.glob(os.path.join(self.save_folder_name, "*.pt"))
            for pt_file in pt_files:
                file_name = os.path.basename(pt_file)
                new_file_name = "fold" + str(times) + "_" + file_name
                dest_path = os.path.join(best_dir, new_file_name)
                shutil.copy2(pt_file, dest_path)