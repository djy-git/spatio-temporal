import os
from os.path import join, dirname, abspath
import sys

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from tabulate import tabulate

from tqdm import tqdm


class PATH:
    root   = abspath(dirname(dirname(__file__)))
    input  = join(root, 'data')
    output = join(root, 'output')
    target = join(root, 'info', 'target.csv')

def impute(data):
    return data.dropna()

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
