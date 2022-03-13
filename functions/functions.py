'''
Modules of functions to auxiliate the project
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os
import sys
import importlib
from datetime import datetime as dti

def mapper_ids(series):
    '''
    Function to mapper hex or other type of id and put it
    in a sequential manner
    Input
        series - (pandas series) a series of pandas with ids
    Output
        series_mapped - (pandas series) the input series mapped
    '''

    map_ = dict()
    
    for j, id in enumerate(series.values):
        map_[id] = int(j)

    # Convert
    series_mapped = series.map(map_)

    return series_mapped, map_