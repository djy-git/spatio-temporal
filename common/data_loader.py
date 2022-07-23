from common.util import *
from common.preprocessing import generate_full_timestamp


def generate_dataset(X, y=None, batch_size=32, shuffle=False):
    """Generate TensorFlow dataset from array-like data (X, y)

    Parameters
    ----------
    X : array-like
        Input data
    y : numpy.ndarray (optional)
        Output data
    batch_size : int (optional)
        Batch size
    shuffle : bool (optional)
        Whether to shuffle data

    Returns
    -------
    TensorFlow dataset : tf.data.Dataset
        Generated dataset
    """
    import tensorflow as tf

    # 1. Convert dtype to float32
    X = np.array(X, dtype=np.float32)
    if y is not None:
        y = np.array(y, dtype=np.float32)

    # 2. Generate TensorFlow dataset
    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(X)
    else:
        ds = tf.data.Dataset.from_tensor_slices((X, y))

    # 3. Options
    if shuffle:
        ds = ds.shuffle(buffer_size=1000)
    return ds.batch(batch_size).cache().prefetch(tf.data.AUTOTUNE)

def split_times(times, in_seq_len, out_seq_len, stride):
    """Split time series into input and output ranges

    Parameters
    ----------
    times : array-like
        Time series data
    in_seq_len : int
        Input sequence length
    out_seq_len : int
        Output sequence length
    stride : int
        Step size between data

    Returns
    -------
    inputs : list
        List of input ranges
    outputs : list
        List of output ranges
    """
    inputs, outputs = [], []
    full_times    = set(range(1, 28800 + 1))  # 28800 = 200일 * 24시간 * 6
    removed_times = full_times - set(times)
    for i in times[::stride]:
        input  = range(i,              i + in_seq_len)
        output = range(i + in_seq_len, i + in_seq_len + out_seq_len)
        if i + in_seq_len + out_seq_len > 28801:
            break
        if len(set(range(i, i + in_seq_len + out_seq_len)) & removed_times) == 0:
            inputs.append(input);  outputs.append(output)
    return inputs, outputs

def make_train_val_test_data(data, in_seq_len, out_seq_len, stride, shuffle, test_size, return_numpy=False, drop_TurbID=True):
    """Generate train, validation, test dataset

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    in_seq_len : int
        Input sequence length
    out_seq_len : int
        Output sequence length
    stride : int
        Step size between data
    shuffle : bool
        Whether to shuffle data
    test_size : int or float
        Number(int) or ratio(float) of test data
    return_numpy : bool (optional)
        Whether to return numpy array
    drop_TurbID : bool (optional)
        Whether to drop TurbID

    Returns
    -------
    train_x : list of pandas.DataFrame or numpy.ndArray
        Input data of training set
    train_y : list of pandas.DataFrame or numpy.ndArray
        Output data of training set
    val_x : list of pandas.DataFrame or numpy.ndArray
        Input data of validation set
    val_y : list of pandas.DataFrame or numpy.ndArray
        Output data of validation set
    test_x : list of pandas.DataFrame or numpy.ndArray
        Input data of test set
    """
    from sklearn.model_selection import train_test_split

    data = copy(data)

    train_x, train_y = [], []
    val_x,   val_y   = [], []
    test_x           = []
    if 'Time' not in data:
        data = generate_full_timestamp(data, drop=True)

    if 'TurbID' not in data:
        data['TurbID'] = 0  # dummy
        drop_TurbID = True

    for i in tqdm(sorted(pd.unique(data['TurbID']))):
        data_tid = data[data['TurbID'] == i]
        if drop_TurbID:
            data_tid.drop(columns=['TurbID'], inplace=True)
        times    = data_tid['Time'].sort_values()
        data_tid = data_tid.set_index('Time')

        # Select time index
        inputs, outputs = split_times(times, in_seq_len, out_seq_len, stride)
        if len(inputs) > 0 and (test_size > 0):
            if shuffle:
                train_in, val_in, train_out, val_out = train_test_split(inputs, outputs, shuffle=True, test_size=test_size)
            else:
                n_val = int(len(inputs) * test_size)
                val_in, val_out = inputs[-n_val:], outputs[-n_val:]
                train_output_threshold = val_out[0].start  # not included

                # Find i s.t. (1 + in_seq_len + i*stride) + out_seq_len = train_output_threshold - 1
                i = ((train_output_threshold - 1) - (1 + in_seq_len) - out_seq_len) // stride
                train_in, train_out = inputs[:i], outputs[:i]

                # for i in range(len(inputs)):
                #     if outputs[i].stop > train_output_threshold:
                #         train_in, train_out = inputs[:i], outputs[:i]
                        # break
        else:
            train_in, val_in, train_out, val_out = inputs, [], outputs, []

        train_x += [data_tid.loc[times] for times in train_in]
        train_y += [data_tid.loc[times] for times in train_out]
        val_x   += [data_tid.loc[times] for times in val_in]
        val_y   += [data_tid.loc[times] for times in val_out]
        test_x  += [data_tid.iloc[-in_seq_len:]]

    if return_numpy:
        train_x, train_y = np.array(train_x, dtype=np.float32), np.array(train_y, dtype=np.float32)
        val_x,   val_y   = np.array(val_x, dtype=np.float32),   np.array(val_y, dtype=np.float32)
        test_x           = np.array(test_x, dtype=np.float32)

    print("* Data Split")
    print("  - Train data(X, y)     :", np.shape(train_x), np.shape(train_y))
    print("  - Validation data(X, y):", np.shape(val_x),   np.shape(val_y))
    print("  - Test data(X)         :", np.shape(test_x))
    return train_x, train_y, val_x, val_y, test_x
