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

    def build_graph(self, train_data, user2tag, item2tag, data_stat):
        print("building graph")
        n_users = data_stat["n_users"]
        n_items = data_stat["n_items"]
        n_tags = data_stat["n_tags"]
        n_nodes = data_stat["n_nodes"]


        train_data = train_data.copy()
        train_data[:,1] = train_data[:,1] + n_users
        
        
        row = train_data[:,0]
        col = train_data[:,1]
        graph_dict = {
            "none" : (row, col)
        }
        if user2tag is not None:
            user2tag = user2tag.copy()
            item2tag = item2tag.copy()
            
            user2tag[:,1] = user2tag[:,1] + n_users + n_items
            item2tag[:,1] = item2tag[:,1] + n_users + n_items
            
            item2tag[:,0] = item2tag[:,0] + n_users
            
            graph_dict["user"] = (np.concatenate([row, user2tag[:,0]]), np.concatenate([col, user2tag[:,1]]))
            graph_dict["item"] = (np.concatenate([row, item2tag[:,0]]), np.concatenate([col, item2tag[:,1]]))
            graph_dict["all"] = (np.concatenate([row, user2tag[:,0], item2tag[:,0]]), np.concatenate([col, user2tag[:,1], item2tag[:,1]]))
        cf_adjs = {}
        norm_mats = {}
        mean_mats = {}
        for k, g in graph_dict.items():
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
        user2tag_file = os.path.join(data_path1, "user2tag.txt")
        item2tag_file = os.path.join(data_path1, "item2tag.txt")

        train_data = pd.read_csv(train_file, sep="\t")[["userID", "itemID"]]

        user2tag, item2tag = None, None
        has_tag = os.path.exists(user2tag_file)
        if has_tag:
            user2tag = pd.read_csv(user2tag_file, sep="\t")
            item2tag = pd.read_csv(item2tag_file, sep="\t")
            user2tag, item2tag = user2tag.to_numpy(), item2tag.to_numpy()

        train_data = train_data.to_numpy()

        test_file = os.path.join(data_path1, "test.txt")
        user2tag_file2 = os.path.join(data_path1, "user2tag_test.txt")
        user2tag_test = pd.read_csv(user2tag_file2, sep="\t")
        user2tag_test = user2tag_test.to_numpy()
        test_data = pd.read_csv(test_file, sep="\t")
        test_data = test_data.to_numpy()
        
        data_stat = self.__stat(train_data, user2tag, item2tag, user2tag_test,test_data)


        user2item_dict = defaultdict(list)
        user2tag_dict = defaultdict(list)
        for idx in range(train_data.shape[0]):
            user2item_dict[train_data[idx, 0]].append(train_data[idx, 1])

        if has_tag:
            for idx in range(user2tag.shape[0]):
                user2tag_dict[user2tag[idx, 0]].append(user2tag[idx, 1])
        data_dict = {
            "user2item": user2item_dict,
            "user2tag": user2tag_dict,
        }

        data_path2 = data_path + '/2'
        train_file = os.path.join(data_path2, "train.txt")
        user2tag_file = os.path.join(data_path2, "user2tag.txt")
        item2tag_file = os.path.join(data_path2, "item2tag.txt")

        train_data2 = pd.read_csv(train_file, sep="\t")[["userID", "itemID"]]

        user2tag2, item2tag2 = None, None
        has_tag = os.path.exists(user2tag_file)
        if has_tag:
            user2tag2 = pd.read_csv(user2tag_file, sep="\t")
            item2tag2 = pd.read_csv(item2tag_file, sep="\t")
            user2tag2, item2tag2 = user2tag2.to_numpy(), item2tag2.to_numpy()

        train_data2 = train_data2.to_numpy()

        test_file = os.path.join(data_path2, "test.txt")
        user2tag_file2 = os.path.join(data_path2, "user2tag_test.txt")
        user2tag_test2 = pd.read_csv(user2tag_file2, sep="\t")
        user2tag_test2 = user2tag_test2.to_numpy()
        test_data2 = pd.read_csv(test_file, sep="\t")
        test_data2 = test_data2.to_numpy()

        return train_data, user2tag, item2tag, data_dict, data_stat, user2tag_test, test_data, train_data2, user2tag2, item2tag2, user2tag_test2, test_data2


    def __stat(self, train_data, user2tag, item2tag, user2tag2,test_data):

        n_users = max(max(train_data[:, 0]),max(test_data[:, 0])) + 1
        n_items = max(train_data[:, 1]) + 1
        n_tags = max(max(user2tag[:, 1]),max(user2tag2[:, 1])) + 1

        n_nodes = n_users + n_items + n_tags

        print(f"n_users:{n_users}")
        print(f"n_items:{n_items}")
        print(f"n_tags:{n_tags}")
        print(f"n_nodes:{n_nodes}")
        print(f"n_interaction:{len(train_data)}")
        if user2tag is not None:
            print(f"n_user2tag:{len(user2tag)}")
            print(f"n_item2tag:{len(item2tag)}")
        return {
            "n_users": n_users,
            "n_items": n_items,
            "n_tags": n_tags,
            "n_nodes": n_nodes
        }


