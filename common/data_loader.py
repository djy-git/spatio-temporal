import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def make_train_data(data, seq_len):
    train_x, train_y = [], []
    for i in tqdm(sorted(pd.unique(data["TurbID"]))):
        tmp_data = data[data["TurbID"] == i]
        for j in range(1, 199 - seq_len):
            # train data ==> 5일 단위
            # label data ==> 2일 단위
            train_days = np.arange(j, j+seq_len)
            label_days = np.arange(j+seq_len, j+seq_len+2)
            train_tmp  = tmp_data[tmp_data.Day.isin(train_days)].drop(columns=['TurbID', 'Day'])
            label_tmp  = tmp_data[tmp_data.Day.isin(label_days)].Patv
            train_x.append(train_tmp)  # (720, 11) = (Tmstamp * days, n_features)
            train_y.append(label_tmp)  # (288)     = (Tmstamp * days)
    train_x, train_y = np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)
    return train_x, train_y

def make_val_data(data, seq_len):
    train_days = np.arange(198-seq_len+1, 199)
    label_days = np.arange(199, 201)
    val_x = data[data.Day.isin(train_days)].drop(columns=['TurbID', 'Day'])
    val_y = data[data.Day.isin(label_days)].Patv
    return np.array(val_x, dtype=np.float32), np.array(val_y, dtype=np.float32)

def make_test_data(data, seq_len):
    train_days = np.arange(200+1-seq_len, 200+1)
    label_days = np.arange(201, 202+1)
    test_x = data[data.Day.isin(train_days)].drop(columns=['TurbID', 'Day'])
    test_y = data[data.Day.isin(label_days)].Patv
    return np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.float32)

def make_train_data_single_turbine(data, seq_len, turbID):
    train_x, train_y = [], []

    tmp_data = data[data["TurbID"] == turbID]
    for j in range(1, 199 - seq_len):
        # train data ==> 5일 단위
        # label data ==> 2일 단위
        train_days = np.arange(j, j+seq_len)
        label_days = np.arange(j+seq_len, j+seq_len+2)
        train_tmp  = tmp_data[tmp_data.Day.isin(train_days)].drop(columns=['TurbID', 'Day'])
        label_tmp  = tmp_data[tmp_data.Day.isin(label_days)].Patv
        train_x.append(train_tmp)  # (720, 11) = (Tmstamp * days, n_features)
        train_y.append(label_tmp)  # (288)     = (Tmstamp * days)
    train_x, train_y = np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)
    return train_x, train_y

def make_val_data_single_turbine(data, seq_len, turbID):
    data = data[data["TurbID"] == turbID]
    train_days = np.arange(198-seq_len+1, 199)
    label_days = np.arange(199, 201)
    val_x = data[data.Day.isin(train_days)].drop(columns=['TurbID', 'Day'])
    val_y = data[data.Day.isin(label_days)].Patv
    return np.array(val_x, dtype=np.float32), np.array(val_y, dtype=np.float32)

def make_test_data_single_turbine(data, seq_len, turbID):
    data = data[data["TurbID"] == turbID]
    train_days = np.arange(200+1-seq_len, 200+1)
    label_days = np.arange(201, 202+1)
    test_x = data[data.Day.isin(train_days)].drop(columns=['TurbID', 'Day'])
    test_y = data[data.Day.isin(label_days)].Patv
    return np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.float32)

def generate_dataset(X, y=None, batch_size=None):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
