from common.util import *


def generate_full_timestamp(data, drop=False):
    data = data.copy()

    # Tmstamp: Timestamp in a day
    tms_list = list(pd.unique(data['Tmstamp']))
    data['Time_in_day'] = data['Tmstamp'].apply(lambda x: tms_list.index(x) + 1)

    for i in data['TurbID'].unique():
        data_tid = data[data['TurbID'] == i]
        if pd.Series(zip(data_tid['Day'], data_tid['Tmstamp'])).is_monotonic_increasing:  # check if sorted
            data.loc[data['TurbID'] == i, 'Time'] = range(1, len(data_tid) + 1)

    # Time: Not duplicated timestamp
    data['Time'] = data['Time'].astype(np.int32)
    if drop:
        data = data.drop(columns=['Day', 'Tmstamp'])
    return data

def marking_data(data, marking_value_target):
    cond = (data['Patv'] <= 0) & (data['Wspd'] > 2.5) | \
           (data['Pab1'] > 89) | (data['Pab2'] > 89) | (data['Pab3'] > 89) | \
           (data['Wdir'] < -180) | (data['Wdir'] > 180) | (data['Ndir'] < -720) | (data['Ndir'] > 720) | \
           (data['Patv'].isnull())
    indices = np.where(~cond)
    data['Patv'].iloc[indices] = marking_value_target
    return data

def impute_data(data):
    data = data[~data['Day'].isin([65, 66, 67])]
    bfill_cols = ['Wspd', 'Wdir', 'Etmp', 'Itmp', 'Ndir', 'Pab1', 'Pab2', 'Pab3', 'Prtv', 'Patv']
    for turbID in data['TurbID'].unique():
        data_tid = data[data['TurbID'] == turbID]
        first_id = data_tid.index[0]
        if data.loc[first_id].isna().any():
           data.loc[first_id, bfill_cols] = data.loc[first_id+1, bfill_cols]
    data = data.interpolate()
    print("Number of Nan values:", sum(data.isna().sum(axis='columns') > 0))
    return data


def preprocess(data):
    temp = data.copy()

    location_data = pd.read_csv("data/turb_location.csv").set_index('TurbID')
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
    temp['Etmp_abs'] = temp['Etmp'] + 243.15

    # Wind absolute direction adjusted Wdir + Ndir
    temp['Wdir_adj'] = np.radians(temp['Wdir'] + temp['Ndir'])
    temp['WdirX'] = np.cos(temp['Wdir_adj'])
    temp['WdirY'] = np.sin(temp['Wdir_adj'])

    # Nacelle Direction cosine sine
    ndir = np.radians(temp['Ndir'] / 180 * np.pi)
    temp['NdirX'] = np.cos(ndir)
    temp['NdirY'] = np.sin(ndir)

    # Wind speed cosine, sine
    wdir = np.radians(temp['Wdir'])
    temp['WspdX'] = temp['Wspd'] * np.cos(wdir)
    temp['WspdY'] = temp['Wspd'] * np.sin(wdir)
    temp['WspdX_abs'] = temp['Wspd'] * np.cos(temp['Wdir_adj'])
    temp['WspdY_abs'] = temp['Wspd'] * np.sin(temp['Wdir_adj'])

    # TSR(Tip speed Ratio)
    alpha = 20
    temp['TSR1'] = 1 / np.tan(np.radians(temp['Pab1'] + alpha))
    temp['TSR2'] = 1 / np.tan(np.radians(temp['Pab2'] + alpha))
    temp['TSR3'] = 1 / np.tan(np.radians(temp['Pab3'] + alpha))

    temp['Bspd1'] = temp['TSR1'] * temp['WspdX']
    temp['Bspd2'] = temp['TSR2'] * temp['WspdX']
    temp['Bspd3'] = temp['TSR3'] * temp['WspdX']

    # RPM derived from blade speed
    temp['Pab'] = ((temp['Pab1'] + temp['Pab2'] + temp['Pab3']) / 3)
    temp['RPM'] = ((temp['Bspd1'] + temp['Bspd2'] + temp['Bspd3']) / 3)
    temp['TSR'] = ((temp['TSR1'] + temp['TSR2'] + temp['TSR3']) / 3)
    #     temp.drop(['TSR1','TSR2','TSR3','Bspd1','Bspd2','Bspd3'], axis=1, inplace=True)

    # Maximum power from wind
    temp['Wspd_cube'] = (temp['WspdX']) ** 3
    temp['P_max'] = ((temp['Wspd']) ** 3) / temp['Etmp_abs']

    # Apparent power
    temp['Papt'] = np.sqrt(temp['Prtv'] ** 2 + temp['Patv'] ** 2)
    temp['Patan'] = np.arctan(temp['Prtv'] / temp['Patv']).fillna(-np.pi / 2)

    ## add 3day & 5day mean value for target according to Hour
    ## average TARGET values of the most recent 3, 5 days
    #     temp['shft1'] = temp['Patv'].shift(144)
    #     temp['shft2'] = temp['Patv'].shift(144 * 2)
    #     temp['shft3'] = temp['Patv'].shift(144 * 3)
    #     temp['shft4'] = temp['Patv'].shift(144 * 4)

    #     temp['avg3'] = np.mean(temp[['Patv', 'shft1', 'shft2']].values, axis=-1)
    #     temp['avg5'] = np.mean(temp[['Patv', 'shft1', 'shft2', 'shft3','shft4']].values, axis=-1)
    #     temp.drop(['shft1','shft2','shft3','shft4'], axis=1, inplace=True)

    #     temp['Patv1'] = temp['Patv'].shift(-144)
    #     temp['Patv2'] = temp['Patv'].shift(-144 * 2)

    #     temp = temp.dropna()
    return temp

