from env import *


class PATH:
    root   = abspath(dirname(dirname(__file__)))
    input  = join(root, 'data')
    output = join(root, 'output')
    target = join(root, 'info', 'target.csv')

def convert_dtypes(data):
    int_cols   = ['TurbID', 'Day']
    float_cols = data.columns.drop(int_cols + ['Tmstamp'])
    data[int_cols]   = data[int_cols].astype(np.int32)
    data[float_cols] = data[float_cols].astype(np.float32)
    return data

def convert_category(data):
    cat_cols = ['TurbID', 'Day']
    data[cat_cols] = data[cat_cols].astype('category')
    return data

