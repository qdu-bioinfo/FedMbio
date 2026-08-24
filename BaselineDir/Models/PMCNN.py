import torch
import torch.nn as nn
import torch.nn.functional as F

class PMCNN_Net(nn.Module):
    def __init__(self, input_len_list):
        super(PMCNN_Net, self).__init__()
        self.branches = nn.ModuleList()
        total_flat_dim = 0
        for length in input_len_list:
            k_size = min(8, length)
            stride = min(7, length)
            if stride < 1: stride = 1
            branch = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=k_size, stride=stride, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 16, kernel_size=k_size, stride=stride, padding=1),
                nn.ReLU()
            )
            self.branches.append(branch)
            dummy = torch.zeros(1, 1, length)
            with torch.no_grad():
                out = branch(dummy)
            total_flat_dim += out.view(1, -1).size(1)
        self.fc1 = nn.Linear(total_flat_dim, 64)
        self.fc2 = nn.Linear(64, 2)

    def forward(self, *inputs):
        branch_outs = []
        for i, x in enumerate(inputs):
            out = self.branches[i](x.unsqueeze(1))
            branch_outs.append(out.view(out.size(0), -1))
        x_cat = torch.cat(branch_outs, dim=1)
        x = F.relu(self.fc1(x_cat))
        return self.fc2(x)