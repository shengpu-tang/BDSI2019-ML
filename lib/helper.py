# EECS 445 - Fall 2017
# Project 1 - helper.py

import pandas as pd
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import train_test_split
import os
import yaml
config = yaml.safe_load(open(os.path.dirname(__file__) + '/config.yaml'))

def load_data(N=2500):
    df_labels = pd.read_csv(open(os.path.dirname(__file__) + '/../data/labels.csv'))[:N]
    IDs = df_labels['RecordID']
    raw_data = {}
    for i in tqdm(IDs, desc='Loading files from disk'):
        raw_data[i] = pd.read_csv('data/files/{}.csv'.format(i))
    return raw_data, df_labels
