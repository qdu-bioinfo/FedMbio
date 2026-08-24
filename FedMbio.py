#!/usr/bin/env python
import torch
import argparse
import os
import warnings
import numpy as np
import random
import time
from FL.servers.serverbio import FedMbio
from FL.utils.result_utils import average_data

warnings.simplefilter("ignore")
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def run(args):
    print(f"--data_type: {args.data_type}")
    print(f"--groups: {args.groups}")
    print(f"--local_learning_rate: {args.local_learning_rate}")
    print(f"--num_clients: {args.num_clients}")
    print(f"--times: {args.times}")
    print(f"--server_epochs: {args.server_epochs}")
    print(f"--local_epochs: {args.local_epochs}")
    print(f"--global_rounds: {args.global_rounds}")
    print(f"--batch_size: {args.batch_size}")

    for i in args.times:
        setup_seed(42)
        print(f"\n============= Running time: {i}th =============")
        args.models = ['MultiModalDiseaseDNN(input_dim)']
        if args.algorithm == "FedMbio":
            server = FedMbio(args, i)
        else:
            raise NotImplementedError
        server.train(times=i)
    average_data(dir=args.save_folder_name_full, times=args.times)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasetDir', type=str, default='datasetDir/',
                        help='Path to read data')
    parser.add_argument('--data_type', type=str, help='Select data type: wgs or 16s')
    parser.add_argument("--groups", type=str, choices=["CTR_CRC", "CTR_ADA"],
                        help="Class pair used for the experiment: CTR_CRC or CTR_ADA")
    parser.add_argument('-nc', "--num_clients", type=int, help="Total number of institutions/clients")
    parser.add_argument('-lr', "--local_learning_rate", type=float, default=0.001, help="Local learning rate")
    parser.add_argument('-jr', "--join_ratio", type=float, default=1.0, help="Ratio of clients per round")
    parser.add_argument('-se', "--server_epochs", type=int, default=20)
    parser.add_argument('-ls', "--local_epochs", type=int, default=5,
                        help="Multiple update steps in one local epoch.")
    parser.add_argument('-gr', "--global_rounds", type=int, default=80)
    parser.add_argument('-lbs', "--batch_size", type=int, default=16)
    parser.add_argument("--save_folder_name", type=str, default='temp')
    parser.add_argument('-fd', "--feature_dim", type=int, default=32)
    parser.add_argument('-t', "--times", type=int, default=[0, 1, 2, 3, 4],
                        help="Running times")
    parser.add_argument('-nb', "--num_classes", type=int, default=2)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedMbio")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    args = parser.parse_args()
    if args.groups == "CTR_CRC":
        args.groups = ["CTR", "CRC"]
    elif args.groups == "CTR_ADA":
        args.groups = ["CTR", "ADA"]
    args.datasetDir = os.path.join(args.datasetDir, args.data_type)
    print(args.datasetDir)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("\ncuda is not avaiable.\n")
        args.device = "cpu"
    args.save_folder_name_full = f'{args.save_folder_name}/{time.time()}/{"_".join(args.groups)}'
    if not os.path.exists(args.save_folder_name_full):
        os.makedirs(args.save_folder_name_full)
    run(args)