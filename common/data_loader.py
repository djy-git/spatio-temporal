import pandas as pd
import numpy as np
from tqdm import tqdm
import tensorflow as tf


def make_train_val_test_data(data, seq_len):
    train_x, train_y = make_train_data(data, seq_len, 198)
    val_x,   val_y   = make_val_data(data, seq_len)
    test_x,  test_y  = make_test_data(data, seq_len)

    train_x = train_x.reshape(-1, train_x[0].shape[0], train_x[0].shape[1])
    val_x   = val_x.reshape(-1, train_x[0].shape[0], train_x[0].shape[1])
    val_y   = val_y.reshape(-1, train_y.shape[-1])
    test_x  = np.array(test_x).reshape(-1, train_x[0].shape[0], train_x[0].shape[1])

    print("Train data(X, y):",      train_x.shape, train_y.shape)
    print("Validation data(X, y):", val_x.shape,   val_y.shape)
    print("Test data(X, y):",       test_x.shape,  test_y.shape)
    return (train_x, train_y), (val_x, val_y), (test_x, test_y)
def make_train_data(data, seq_len, n_train):
    """
    Last training set: [n_train-2-seq_len:n_train-2]
    Last test set    : [n_train-2:n_train]
    """
    train_x, train_y = [], []
    for i in tqdm(sorted(pd.unique(data["TurbID"]))):
        tmp_data = data[data["TurbID"] == i]
        for j in range(1, (n_train-2) - (seq_len-1) + 1):  # 2일 test
            # train data ==> seq_len일 단위
            # label data ==> 2일 단위
            train_days = np.arange(j, j+seq_len)
            label_days = np.arange(j+seq_len, j+seq_len+2)
            if (tmp_data.Day.isin(np.arange(j, j+seq_len+2)).sum() / 144) == (seq_len+2):  # 1Day = 10m * 144
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
    if label_days[0] in data.Day:
        test_y = data[data.Day.isin(label_days)].Patv
    else:
        test_y = np.array([])  # dummy
    return np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.float32)

def make_train_val_test_data_single_turbine(data, seq_len):
    train_xs, train_ys = [], []
    val_xs,   val_ys   = [], []
    test_xs,  test_ys  = [], []

    for turbID in tqdm(sorted(data['TurbID'].unique())):
        train_x, train_y = make_train_data_single_turbine(data, seq_len, 198, turbID)
        val_x,   val_y   = make_val_data_single_turbine(data, seq_len, turbID)
        test_x,  test_y  = make_test_data_single_turbine(data, seq_len, turbID)

        train_x = train_x.reshape(-1, train_x[0].shape[0], train_x[0].shape[1])
        val_x = val_x.reshape(-1, train_x[0].shape[0], train_x[0].shape[1])
        val_y = val_y.reshape(-1, train_y.shape[-1])
        test_x = np.array(test_x).reshape(-1, train_x[0].shape[0], train_x[0].shape[1])
        if len(test_y) > 0:
            test_y = test_y.reshape(-1, train_y.shape[-1])

        train_xs.append(train_x);  train_ys.append(train_y)
        val_xs.append(val_x);      val_ys.append(val_y)
        test_xs.append(test_x);    test_ys.append(test_y)

    train_xs, train_ys = np.array(train_xs), np.array(train_ys)
    val_xs,   val_ys   = np.array(val_xs),   np.array(val_ys)
    test_xs,  test_ys  = np.array(test_xs),  np.array(test_ys)

    print("Train data(X, y):", train_xs.shape, train_ys.shape)
    print("Validation data(X, y):", val_xs.shape, val_ys.shape)
    print("Test data(X, y):", test_xs.shape, test_ys.shape)
    return (train_xs, train_ys), (val_xs, val_ys), (test_xs, test_ys)
def make_train_data_single_turbine(data, seq_len, n_train, turbID):
    train_x, train_y = [], []
    tmp_data = data[data["TurbID"] == turbID]
    for j in range(1, (n_train - 2) - (seq_len - 1) + 1):  # 2일 test
        # train data ==> seq_len일 단위
        # label data ==> 2일 단위
        train_days = np.arange(j, j + seq_len)
        label_days = np.arange(j + seq_len, j + seq_len + 2)
        train_tmp = tmp_data[tmp_data.Day.isin(train_days)].drop(columns=['TurbID', 'Day'])
        label_tmp = tmp_data[tmp_data.Day.isin(label_days)].Patv
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
    if label_days[0] in data.Day:
        test_y = data[data.Day.isin(label_days)].Patv
    else:
        test_y = np.array([])  # dummy
    return np.array(test_x, dtype=np.float32), np.array(test_y, dtype=np.float32)


# `make_train_data()`과 교체 대기, 수정 가능
def make_train_data_10min(data, seq_len):
    train_x, train_y = [], []
    for i in tqdm(sorted(pd.unique(data["TurbID"]))):
        tmp_data = data[data["TurbID"] == i]
        train_x, train_y = [], []
        window_offset = 0
        window_interval = 1 # 1 : 10min, 6 : 60min
        train_window_size = 144 * seq_len
        label_window_size = 144 * 2
        while (window_offset + train_window_size + label_window_size <= len(tmp_data)):
            train_tmp = tmp_data[window_offset:
                                 window_offset + train_window_size].drop(columns=['TurbID', 'Day'])
            label_tmp = tmp_data[window_offset + train_window_size:
                                 window_offset + train_window_size + label_window_size].Patv
            window_offset += window_interval
            train_x.append(train_tmp)
            train_y.append(label_tmp)
    train_x, train_y = np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)
    return train_x, train_y


def generate_dataset(X, y=None, batch_size=None):
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)
