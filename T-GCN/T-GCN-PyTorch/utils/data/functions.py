import numpy as np
import pandas as pd
import torch
import os
import os.path as osp



def load_features(feat_path, year, dtype=np.float32):
    dirpath = os.path.dirname(feat_path)

    feat_path = os.path.join(dirpath, str(year)+".npz")
    
    feat_df = pd.read_csv(feat_path)
    feat = np.array(feat_df, dtype=dtype)
    return feat


# def load_adjacency_matrix(adj_path, dtype=np.float32):
#     adj_df = pd.read_csv(adj_path, header=None)
#     adj = np.array(adj_df, dtype=dtype)
#     return adj

def load_adjacency_matrix(adj_path, year, dtype=np.float32):
    adj = np.load(osp.join(adj_path, str(year)+"_adj.npz"))["x"]
    return adj


def generate_dataset(
    data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
):
    """
    :param data: feature matrix
    :param seq_len: length of the train data sequence
    :param pre_len: length of the prediction data sequence
    :param time_len: length of the time series in total
    :param split_ratio: proportion of the training set
    :param normalize: scale the data to (0, 1], divide by the maximum value in the data
    :return: train set (X, Y) and test set (X, Y)
    """
    if time_len is None:
        time_len = data.shape[0]
    if normalize:
        max_val = np.max(data)
        data = data / max_val
    train_size = int(time_len * split_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:time_len]
    train_X, train_Y, test_X, test_Y = list(), list(), list(), list()
    for i in range(len(train_data) - seq_len - pre_len):
        train_X.append(np.array(train_data[i : i + seq_len]))
        train_Y.append(np.array(train_data[i + seq_len : i + seq_len + pre_len]))
    for i in range(len(test_data) - seq_len - pre_len):
        test_X.append(np.array(test_data[i : i + seq_len]))
        test_Y.append(np.array(test_data[i + seq_len : i + seq_len + pre_len]))
    return np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)


# def generate_torch_datasets(
#     data, seq_len, pre_len, time_len=None, split_ratio=0.8, normalize=True
# ):
#     train_X, train_Y, test_X, test_Y = generate_dataset(
#         data,
#         seq_len,
#         pre_len,
#         time_len=time_len,
#         split_ratio=split_ratio,
#         normalize=normalize,
#     )

#     #print(train_X.shape)         #(num_data, seq_len, N)
#     #print(train_Y.shape)         #(num_data, pre_len, N)
#     train_dataset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(train_X), torch.FloatTensor(train_Y)
#     )
#     test_dataset = torch.utils.data.TensorDataset(
#         torch.FloatTensor(test_X), torch.FloatTensor(test_Y)
#     )
#     return train_dataset, test_dataset


def generate_torch_datasets(data_path, year):

    dirpath = os.path.dirname(data_path)

    filename = os.path.join(dirpath, str(year)+"_30day.npz")

    print('load file:', filename)

    file_data = np.load(filename, allow_pickle=True)
    

    train_x = file_data['train_x']                              # (num_data, seq_len, num_node)
    max_train_x = np.max(train_x)
    train_y = file_data['train_y']                              # (num_data, seq_len, num_node)
    max_train_y = np.max(train_y)


    # val_x = file_data['val_x']                                  # (num_data, seq_len, num_node)
    # val_target = file_data['val_y']                             # (num_data, seq_len, num_node)

    test_x = file_data['test_x']                                # (num_data, seq_len, num_node)
    max_test_x = np.max(test_x)
    test_y = file_data['test_y']                                # (num_data, seq_len, num_node)
    max_test_y = np.max(test_y)

    #print(train_X.shape)         #(num_data, seq_len, N)
    #print(train_Y.shape)         #(num_data, pre_len, N)
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(train_x), torch.FloatTensor(train_y)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(test_x), torch.FloatTensor(test_y)
    )
    return train_dataset, test_dataset, max(max_train_x, max_train_y, max_test_x, max_test_y)