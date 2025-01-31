import torch
import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from collections import defaultdict
import random
import scipy.sparse as sp
def _bi_norm_lap(adj):
    rowsum = np.array(adj.sum(1))

    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)

    bi_lap = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
    return bi_lap.tocoo()

def _si_norm_lap(adj):
    rowsum = np.array(adj.sum(1))

    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat_inv = sp.diags(d_inv)

    norm_adj = d_mat_inv.dot(adj)
    return norm_adj.tocoo()

class DataUtils():
    def __init__(self, args):

        super(DataUtils, self).__init__()
        self.args = args
        self.device = torch.device(f"cuda:{self.args.gpu_id}")

    def build_graph(self, train_data, visit2icd, ccs2icd, data_stat):
        print("building graph")
        n_visits = data_stat["n_visits"]
        n_ccss = data_stat["n_ccss"]
        n_icds = data_stat["n_icds"]
        n_nodes = data_stat["n_nodes"]


        train_data = train_data.copy()
        train_data[:,1] = train_data[:,1] + n_visits
        
        
        row = train_data[:,0]
        col = train_data[:,1]
        graph_dict = {
            "none" : (row, col)
        }
        if visit2icd is not None:
            visit2icd = visit2icd.copy()
            ccs2icd = ccs2icd.copy()
            
            visit2icd[:,1] = visit2icd[:,1] + n_visits + n_ccss
            ccs2icd[:,1] = ccs2icd[:,1] + n_visits + n_ccss
            
            ccs2icd[:,0] = ccs2icd[:,0] + n_visits
            
            graph_dict["visit"] = (np.concatenate([row, visit2icd[:,0]]), np.concatenate([col, visit2icd[:,1]]))
            graph_dict["ccs"] = (np.concatenate([row, ccs2icd[:,0]]), np.concatenate([col, ccs2icd[:,1]]))
            graph_dict["all"] = (np.concatenate([row, visit2icd[:,0], ccs2icd[:,0]]), np.concatenate([col, visit2icd[:,1], ccs2icd[:,1]]))
        cf_adjs = {}
        norm_mats = {}
        mean_mats = {}
        for k, g in graph_dict.ccss():
            row, col = g
            if self.args.inverse_r:
                row_t = np.concatenate([row, col])
                col_t = np.concatenate([col, row])
                row = row_t
                col = col_t
            idx = np.unique(np.stack([row, col]), axis=1)
            vals = [1.] * (idx.shape[1])
            
            cf_adj = sp.coo_matrix((vals, idx), shape=(n_nodes, n_nodes))
            norm_mat = _bi_norm_lap(cf_adj)
            mean_mat = _si_norm_lap(cf_adj)
            cf_adjs[k] = cf_adj
            norm_mats[k] = norm_mat
            mean_mats[k] = mean_mat

        return cf_adjs, norm_mats, mean_mats


    def read_files(self):
        print("reading files ...")
        data_path = os.path.join(self.args.data_path, self.args.dataset)
        data_path1 = data_path + '/1'
        train_file = os.path.join(data_path1, "train.txt")
        visit2icd_file = os.path.join(data_path1, "visit2icd.txt")
        ccs2icd_file = os.path.join(data_path1, "ccs2icd.txt")

        train_data = pd.read_csv(train_file, sep="\t")[["visitID", "ccsID"]]

        visit2icd, ccs2icd = None, None
        has_icd = os.path.exists(visit2icd_file)
        if has_icd:
            visit2icd = pd.read_csv(visit2icd_file, sep="\t")
            ccs2icd = pd.read_csv(ccs2icd_file, sep="\t")
            visit2icd, ccs2icd = visit2icd.to_numpy(), ccs2icd.to_numpy()

        train_data = train_data.to_numpy()

        test_file = os.path.join(data_path1, "test.txt")
        visit2icd_file2 = os.path.join(data_path1, "visit2icd_test.txt")
        visit2icd_test = pd.read_csv(visit2icd_file2, sep="\t")
        visit2icd_test = visit2icd_test.to_numpy()
        test_data = pd.read_csv(test_file, sep="\t")
        test_data = test_data.to_numpy()
        
        data_stat = self.__stat(train_data, visit2icd, ccs2icd, visit2icd_test,test_data)


        visit2ccs_dict = defaultdict(list)
        visit2icd_dict = defaultdict(list)
        for idx in range(train_data.shape[0]):
            visit2ccs_dict[train_data[idx, 0]].append(train_data[idx, 1])

        if has_icd:
            for idx in range(visit2icd.shape[0]):
                visit2icd_dict[visit2icd[idx, 0]].append(visit2icd[idx, 1])
        data_dict = {
            "visit2ccs": visit2ccs_dict,
            "visit2icd": visit2icd_dict,
        }

        data_path2 = data_path + '/2'
        train_file = os.path.join(data_path2, "train.txt")
        visit2icd_file = os.path.join(data_path2, "visit2icd.txt")
        ccs2icd_file = os.path.join(data_path2, "ccs2icd.txt")

        train_data2 = pd.read_csv(train_file, sep="\t")[["visitID", "ccsID"]]

        visit2icd2, ccs2icd2 = None, None
        has_icd = os.path.exists(visit2icd_file)
        if has_icd:
            visit2icd2 = pd.read_csv(visit2icd_file, sep="\t")
            ccs2icd2 = pd.read_csv(ccs2icd_file, sep="\t")
            visit2icd2, ccs2icd2 = visit2icd2.to_numpy(), ccs2icd2.to_numpy()

        train_data2 = train_data2.to_numpy()

        test_file = os.path.join(data_path2, "test.txt")
        visit2icd_file2 = os.path.join(data_path2, "visit2icd_test.txt")
        visit2icd_test2 = pd.read_csv(visit2icd_file2, sep="\t")
        visit2icd_test2 = visit2icd_test2.to_numpy()
        test_data2 = pd.read_csv(test_file, sep="\t")
        test_data2 = test_data2.to_numpy()

        return train_data, visit2icd, ccs2icd, data_dict, data_stat, visit2icd_test, test_data, train_data2, visit2icd2, ccs2icd2, visit2icd_test2, test_data2


    def __stat(self, train_data, visit2icd, ccs2icd, visit2icd2,test_data):

        n_visits = max(max(train_data[:, 0]),max(test_data[:, 0])) + 1
        n_ccss = max(train_data[:, 1]) + 1
        n_icds = max(max(visit2icd[:, 1]),max(visit2icd2[:, 1])) + 1

        n_nodes = n_visits + n_ccss + n_icds

        print(f"n_visits:{n_visits}")
        print(f"n_ccss:{n_ccss}")
        print(f"n_icds:{n_icds}")
        print(f"n_nodes:{n_nodes}")
        print(f"n_interaction:{len(train_data)}")
        if visit2icd is not None:
            print(f"n_visit2icd:{len(visit2icd)}")
            print(f"n_ccs2icd:{len(ccs2icd)}")
        return {
            "n_visits": n_visits,
            "n_ccss": n_ccss,
            "n_icds": n_icds,
            "n_nodes": n_nodes
        }


