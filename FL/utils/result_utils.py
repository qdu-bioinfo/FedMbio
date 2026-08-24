import h5py
import numpy as np
import os
def average_data(dir, times):
    test_acc, test_auc = get_all_results_for_one_algo(dir, times)
    auc_list = []
    corresponding_acc_for_auc = []
    for i in times:
        current_auc = test_auc[i]
        current_acc = test_acc[i]
        auc_idx = np.argmax(current_auc)
        auc = current_auc[auc_idx]
        corresponding_acc = current_acc[auc_idx]
        auc_list.append(auc)
        corresponding_acc_for_auc.append(corresponding_acc)
    mean_auc = np.mean(auc_list)
    mean_corresponding_acc_for_auc = np.mean(corresponding_acc_for_auc)
    print(f"\n\n{len(times)}-fold average results:")
    print(f"AUC: {mean_auc:.4f}")
    print(f"ACC: {mean_corresponding_acc_for_auc:.4f}")

def get_all_results_for_one_algo(dir, times):
    test_auc = [[] for _ in range(5)]
    test_acc = [[] for _ in range(5)]
    for i in times:
        file_name = dir + f"/Result_{i}.h5"
        test_acc[i] = (np.array(read_data_then_delete(file_name, seek='acc', delete=False)))
        test_auc[i] = (np.array(read_data_then_delete(file_name, seek='auc', delete=True)))
    return test_acc, test_auc

def read_data_then_delete(file_name, seek, delete=False):
    with h5py.File(file_name, 'r') as hf:
        if (seek == 'acc'):
            out = np.array(hf.get('rs_test_acc'))
        elif (seek == 'auc'):
            out = np.array(hf.get('rs_test_auc'))
        else:
            raise ValueError(f"Invalid seek value: '{seek}'. Expected 'acc' or 'auc'.")
    if delete:
        os.remove(file_name)
    return out