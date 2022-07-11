from common.util import *


def impute_data(data):
    data = data[~data['Day'].isin([65, 66, 67])]
    for turbID in data['TurbID'].unique():
        data_tid = data[data['TurbID'] == turbID]
        first_id = data_tid.index[0]
        if data.loc[first_id].isna().any():
           data.loc[first_id, data.columns[2:]] = data.loc[first_id+1, data.columns[2:]]
    data = data.interpolate()
    print("Number of Nan values:", sum(data.isna().sum(axis='columns') > 0))
    return data

