import numpy as np

from common.util import *


def generate_full_timestamp(data, drop=False):
    """Generate not duplicated time series index

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    drop : bool (optional)
        Whether to drop `Day`, `Tmstamp` features in returned data

    Returns
    -------
    data : pandas.DataFrame
        Data which has Time indicator features `Time`
    """
    data = data.copy()

    for i in data['TurbID'].unique():
        data_tid = data[data['TurbID'] == i]
        if pd.Series(zip(data_tid['Day'], data_tid['Tmstamp'])).is_monotonic_increasing:  # check if sorted
            data.loc[data['TurbID'] == i, 'Time'] = range(1, len(data_tid) + 1)

    # Time: Not duplicated timestamp
    data['Time'] = data['Time'].astype(np.int32)
    if drop:
        data = data.drop(columns=['Day', 'Tmstamp'])
    return data


def get_idxs_mark(data):
    cond = (data['Patv'] <= 0) & (data['Wspd'] > 2.5) | \
           (data['Pab1'] > 89) | (data['Pab2'] > 89) | (data['Pab3'] > 89) | \
           (data['Wdir'] < -180) | (data['Wdir'] > 180) | (data['Ndir'] < -720) | (data['Ndir'] > 720) | \
           (data['Patv'].isnull())
    return np.where(cond)[0]


def marking_data(data, marking_value_target):
    data = copy(data)
    indices = get_idxs_mark(data)
    data['Patv'].iloc[indices] = marking_value_target
    return data


def impute_data(data, threshold=6 * 12):
    """Impute data
    1. Drop continuous missing rows (more than threshold)
    2. Fill missing rows with backward values using threshold
    3. Interpolation

    Parameters
    ----------
    data : pandas.DataFrame
        Input data

    threshold : int (optional)
        Threshold of continuous missing values

    Returns
    -------
    data_imp : pandas.DataFrame
        Imputed data
    """
    data_imp = pd.DataFrame()
    for turbID in tqdm(data['TurbID'].unique()):
        data_tid = data[data['TurbID'] == turbID]
        idxs = (data_tid.isna().sum(axis='columns') > 0)
        idxs_nan = idxs[idxs].index
        idxs_removed = []

        if len(idxs_nan) > 0:
            s, e = 0, 1
            while e < len(idxs_nan):
                cur = idxs_nan[s:e]
                if idxs_nan[e] == cur[-1] + 1:
                    e += 1
                else:
                    if len(cur) >= threshold:
                        idxs_removed += list(cur)
                    s = e
                    e = s + 1
            else:
                if len(cur) >= threshold:
                    idxs_removed += list(cur)

        data_tid_imp = data_tid.drop(idxs_removed).interpolate().fillna(method='bfill')
        data_imp = data_imp.append(data_tid_imp)
    check_nan(data_imp, "Imputing")
    return data_imp


def feature_engineering(data, encode_TurbID=False, compute_Pmax_method='simple', compute_Pmax_clipping=True, power_constant=0.5):
    """Add features with feature engineering

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    encode_TurbID: bool (optional)
        Whether to encode TurbID with Binary encoding
    compute_Pmax_method : str (optional)
        Method to compute Maximum Power(Pmax)
    compute_Pmax_clipping : bool (optional)
        Whether to clip Pmax with Patv range
    power_constant : float (optional)

    Returns
    -------
    data : pandas.DataFrame
        Preprocessed data
    """
    temp = data.copy()

    ## Binary encoding of TurbID
    if encode_TurbID:
        from category_encoders import BinaryEncoder
        temp = BinaryEncoder().fit_transform(temp['TurbID'].astype(str)).join(temp)

    location_data = pd.read_csv(join(PATH.input, "turb_location.csv")).set_index('TurbID')
    x_grid = np.linspace(0, 5000, 6)
    y_grid = np.linspace(0, 12000, 25)
    location_data['x'] = location_data['x'].apply(lambda x: x_grid[abs(x_grid - x).argsort()[0]])
    location_data['y'] = location_data['y'].apply(lambda y: y_grid[abs(y_grid - y).argsort()[0]])
    location_dict = location_data.to_dict('index')
    temp['locX'] = temp['TurbID'].apply(lambda x: location_dict[x]['x'])
    temp['locY'] = temp['TurbID'].apply(lambda y: location_dict[y]['y'])

    ## add cyclical encoded time feature
    temp.Tmstamp = temp.Tmstamp.apply(lambda x: int(x.split(':')[0]) + int(x.split(':')[1]) / 60)
    temp['TimeX'] = np.cos(2 * np.pi * (temp.Tmstamp / 24))
    temp['TimeY'] = np.sin(2 * np.pi * (temp.Tmstamp / 24))

    ## add cyclical encoded time feature
    temp['DayX'] = np.cos(2 * np.pi * (temp.Day / 365.2425))
    temp['DayY'] = np.sin(2 * np.pi * (temp.Day / 365.2425))

    # celsius to kelvin
    c = 243.15
    temp['Etmp_abs'] = temp['Etmp'] + c

    # Nacelle Direction cosine sine
    temp['Ndir_cos'] = np.cos(temp['Ndir'] / 180 * np.pi)
    temp['Ndir_sin'] = np.sin(temp['Ndir'] / 180 * np.pi)
    temp['Wdir_adj'] = np.radians(temp['Wdir'] + temp['Ndir'])
    temp['WdirX'] = np.cos(temp['Wdir_adj'])
    temp['WdirY'] = np.sin(temp['Wdir_adj'])

    # Nacelle Direction cosine sine
    ndir = np.radians(temp['Ndir'] / 180 * np.pi)
    temp['NdirX'] = np.cos(ndir)
    temp['NdirY'] = np.sin(ndir)

    # Wind speed should be positive for computing Pmax
    vals         = temp['Wspd'].value_counts().index
    min_val      = vals[vals > 0][0]
    temp['Wspd'] = temp['Wspd'].clip(min_val, max(temp['Wspd']))

    # Wind speed cosine, sine
    wdir = np.radians(temp['Wdir'])
    temp['WspdX'] = temp['Wspd'] * np.cos(wdir)
    temp['WspdY'] = temp['Wspd'] * np.sin(wdir)
    temp['WspdX_abs'] = temp['Wspd'] * np.cos(temp['Wdir_adj'])
    temp['WspdY_abs'] = temp['Wspd'] * np.sin(temp['Wdir_adj'])

    # TSR(Tip speed Ratio)
    alpha = 40
    temp['TSR1'] = 1 / np.tan(np.radians((temp['Pab1'] + alpha).apply(lambda x: min(x, 89)))).apply(lambda x: max(x, 0))
    temp['TSR2'] = 1 / np.tan(np.radians((temp['Pab2'] + alpha).apply(lambda x: min(x, 89)))).apply(lambda x: max(x, 0))
    temp['TSR3'] = 1 / np.tan(np.radians((temp['Pab3'] + alpha).apply(lambda x: min(x, 89)))).apply(lambda x: max(x, 0))

    temp['Bspd1'] = temp['TSR1'] * temp['WspdX']
    temp['Bspd2'] = temp['TSR2'] * temp['WspdX']
    temp['Bspd3'] = temp['TSR3'] * temp['WspdX']

    # RPM derived from blade speed
    temp['Pab'] = ((temp['Pab1'] + temp['Pab2'] + temp['Pab3']) / 3)
    temp['TSR'] = ((temp['TSR1'] + temp['TSR2'] + temp['TSR3']) / 3)
    temp['RPM'] = ((temp['Bspd1'] + temp['Bspd2'] + temp['Bspd3']) / 3)
    # temp.drop(['TSR1','TSR2','TSR3','Bspd1','Bspd2','Bspd3'], axis=1, inplace=True)

    # Maximum power from wind
    temp['Wspd_cube'] = temp['WspdX'] ** 3
    temp['Pmax'] = compute_Pmax(temp, method=compute_Pmax_method, clipping=compute_Pmax_clipping, power_constant=power_constant)

    # Apparent power, Power arctangent
    temp['Papt']  = np.sqrt(temp['Prtv'] ** 2 + temp['Patv'] ** 2)
    temp['Patan'] = np.arctan(temp['Prtv'] / temp['Patv']).fillna(-np.pi / 2)

    # shifted weather features
    #shifted 5days
    temp['Wspd5'] = temp['Wspd'].shift(144*5)
    temp['Patv5'] = temp['Patv'].shift(144*5)
    temp['Etmp5'] = temp['Etmp'].shift(144*5)
    temp['Etmp4'] = temp['Etmp'].shift(144*4)

    check_nan(temp, "Feature engineering")
    return temp


def select_features(data, threshold=0.4):
    """Select features which has correlations with Patv more than threshold

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    threshold : float (optional)
        Correlation threshold

    Returns
    -------
    features : list
        Relevant features
    """
    corr = data.corr()['Patv'].sort_values()
    corr_abs = corr.abs().sort_values()
    cols = list(corr_abs[corr_abs > threshold].index)
    if include_TurbID:
        cols = [col for col in data if col.startswith('TurbID_')] + [col for col in cols if 'TurbID' not in col]
    print("* Selected features:", cols)
    return cols


def smooth(data, target, window_length=11, polyorder=3):
    """Smooth target using [Savitzky-Golay filter](https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.savgol_filter.html)

    Parameter
    ---------
    data : pandas.DataFrame
        Input data
    target : str
        Smoothing feature
    window_length : int (optional)
        Window length for smoothing (the bigger, the smoother)
    polyorder : int (optional)
        Order of prediction function for smoothing (the bigger, the smoother)

    Returns
    -------
    data_sm : pandas.DataFrame
    """
    from scipy.signal import savgol_filter

    data_mark = copy(data)
    data_mark = marking_data(data_mark, None)  # Except anomaly from computations

    data_sm = data_mark.interpolate()  # Alleviate anomaly impact
    for turbID in data_sm['TurbID'].unique():
        data_tid = data_sm[data_sm['TurbID'] == turbID]
        data_sm.loc[data_tid.index, target] = savgol_filter(data_tid[target], window_length, polyorder)
    check_nan(data_sm, "Smoothing")
    return data_sm


def positional_encoding(seq_len, d, n=10000):
    P = np.zeros((seq_len, d))
    for k in range(seq_len):
        for i in np.arange(d // 2):
            denom = np.power(n, 2 * i / d)
            P[k, 2 * i] = np.sin(k / denom)
            P[k, 2 * i + 1] = np.cos(k / denom)
    return P


def scale(data, scaler):
    if not isinstance(data, np.ndarray):
        data = np.array(data, dtype=np.float32)
    if len(data.shape) == 3:
        return scaler.transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    elif len(data.shape) == 2:
        return scaler.transform(data)
    else:
        raise ValueError("len(data.shape) should be 2 or 3")
def inverse_scale(data, scaler):
    if len(data.shape) == 3:
        return scaler.inverse_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)
    elif len(data.shape) == 2:
        return scaler.inverse_transform(data)
    else:
        raise ValueError("len(data.shape) should be 2 or 3")


# Outlier handler for multiple columns
def outlier_handler(data, columns, window_length=21, polyorder=3, verbose=False, smooth=False):
    data = data.copy()
    window_size = 2
    for i in tqdm(data['Day'].unique()):
        temp = data[(data['Day'] >= i) & (data['Day'] <= i + window_size - 1)].copy()
        if verbose:
            print('Day ', i)
        temp = drop_outliers(temp, columns, verbose)
        temp = fill_gaps(temp, columns)
        if smooth:
            temp = curve_fit(temp, columns, window_length=window_length, polyorder=polyorder)
        data[(data['Day'] >= i) & (data['Day'] <= i + window_size - 1)] = temp
    return data


def drop_outliers(data, columns, verbose=False):
    temp = data.copy()
    for column in columns:
        temp[f'{column}_diff'] = temp[column].diff(1)
        temp[f'{column}_diff'] = temp[column].diff(1)

    q1 = pd.DataFrame(temp).quantile(0.25)
    q3 = pd.DataFrame(temp).quantile(0.75)
    iqr = q3 - q1  # Interquartile range
    fence_low = q1 - (1.5 * iqr)
    fence_high = q3 + (1.5 * iqr)

    for column in columns:
        if verbose:
            print(
                f"      {column} Fence High/Low ({fence_high[column]}/{fence_low[column]}), Count: {temp.loc[(temp[column] < fence_low[column]) | (temp[column] > fence_high[column]), column].count()}")
        temp.loc[temp[column] > fence_high[column], column] = np.nan
        temp.loc[temp[column] < fence_low[column], column] = np.nan
        if verbose:
            print(
                f"      {f'{column}_diff'} Fence High/Low ({fence_high[f'{column}_diff']}/{fence_low[f'{column}_diff']}), Count: {temp.loc[(temp[f'{column}_diff'] < fence_low[f'{column}_diff']) | (temp[f'{column}_diff'] > fence_high[f'{column}_diff']), f'{column}_diff'].count()}")
        temp.loc[temp[f'{column}_diff'] > fence_high[f'{column}_diff'], column] = np.nan
        temp.loc[temp[f'{column}_diff'] < fence_low[f'{column}_diff'], column] = np.nan
        temp.drop([f'{column}_diff'], axis=1, inplace=True)
    return temp


def fill_gaps(data, columns):
    temp = data.copy()
    for column in columns:
        temp[column] = temp[column].interpolate()
        temp[column] = temp[column].fillna(method='bfill')
    return temp


def curve_fit(data, columns, window_length=21, polyorder=3):
    from scipy.signal import savgol_filter

    temp = data.copy()
    for column in columns:
        temp[column] = savgol_filter(temp[column], window_length, polyorder)
    return temp


def compute_Pmax_constants(data, method='simple', power_constant=0.5):
    """Compute constants for computing maximum Power(Pmax)

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    method : str (optional)
        Method to compute power constant
    power_constant : float (optional)

    Returns
    -------
    constants : dict
    """
    data = copy(data)

    # Prepare necessary features
    if 'Wspd_cube' not in data:
        if 'WspdX' not in data:
            data['WspdX'] = data['Wspd'] * np.cos(np.radians(data['Wdir']))
        data['Wspd_cube'] = data['WspdX'] ** 3
    if 'Etmp_abs' not in data:
        data['Etmp_abs'] = data['Etmp'] + 243.15

    constants = {turbID: None for turbID in data['TurbID'].unique()}
    for turbID in data['TurbID'].unique():
        d = data[data['TurbID'] == turbID]
        if method == 'simple':
            constants[turbID] = (d['Patv'] / (d['Wspd_cube'] / d['Etmp_abs'])).mean()
        elif method == 'clipping':
            constants[turbID] = (d['Patv'] / (d['Wspd_cube'] / d['Etmp_abs'])).clip(0, 807*power_constant).mean()
        else:
            raise ValueError(f"{method} should be in ['simple', 'clipping']")
    return constants


def compute_Pmax(data, method='simple', clipping=True, clipping_min_val=None, clipping_max_val=None, constants=None, power_constant=0.5):
    """Compute Maximum Power(Pmax)

    Parameters
    ----------
    data : pandas.DataFrame
        Input data
    method : str (optional)
        Method to compute power constant
    clipping : bool (optional)
        Whether to clip Pmax with Patv range
    clipping_min_val : float (optional)
        Min value for clipping
    clipping_max_val : float (optional)
        Max value for clipping
    constants : dict (optional)
        Computed Pmax constants
    power_constant : float (optional)

    Returns
    -------
    Pmax : pandas.Series
        Maximum power
    """
    data = copy(data)

    # Ignore anomaly
    try:
        data_comp = copy(data)
        data_comp = marking_data(data, None).dropna()  # Except anomaly from computations
    except:
        pass  # Assume that data is already cleaned

    # Prepare necessary features
    if 'Wspd_cube' not in data:
        if 'WspdX' not in data:
            data_comp['WspdX'] = data_comp['Wspd'] * np.cos(np.radians(data_comp['Wdir']))
        data_comp['Wspd_cube'] = data_comp['WspdX'] ** 3
    if 'Etmp_abs' not in data:
        data_comp['Etmp_abs'] = data_comp['Etmp'] + 243.15

    # Compute constants
    if constants is None:
        constants = compute_Pmax_constants(data_comp, method, power_constant)

    # Compute Pmax
    for turbID, C in constants.items():
        data_comp.loc[data['TurbID'] == turbID, 'C'] = C
    data['Pmax'] = data_comp['C'] * (data_comp['Wspd_cube'] / data_comp['Etmp_abs'])
    if clipping:
        if clipping_min_val is None:
            clipping_min_val = min(data['Patv'])
        if clipping_max_val is None:
            clipping_max_val = max(data['Patv'])
        data['Pmax'] = data['Pmax'].clip(clipping_min_val, clipping_max_val)
    return data['Pmax']

# def select_residual_threshold(true, pred, plot=False, n_rows=10, clip_Pmax=True):
#     from matplotlib.cbook import boxplot_stats
#
#     if clip_Pmax:
#         pred = np.clip(pred, min(true), max(true))
#     res = true - pred
#     threshold = boxplot_stats(res)[0]['whislo']
#
#     if plot:
#         n_data     = len(true)
#         batch_size = (n_data // n_rows) if n_data % n_rows == 0 else (n_data // n_rows + 1)
#         fig, axes = plt.subplots(n_rows, 1, figsize=(40, n_rows*2))
#         for i, ax in enumerate(axes.flatten()):
#             res_batch  = res[i*batch_size:(i+1)*batch_size]
#             ax.plot(res_batch)
#             ax.axhline(0, color='k')
#             ax.axhline(threshold, color='b')
#             ax.set_xticklabels([])
#         fig.tight_layout()
#         plt.show()
#
#     return threshold