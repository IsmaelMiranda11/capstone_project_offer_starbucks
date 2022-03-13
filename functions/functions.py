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
from statsmodels.stats import proportion

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


def test_proportions_in_dataframe(df, column, cat1,cat2, metric, return_=False):
    '''
    Function to assert that two proportions are statistically different
    '''
    serie1 = df.loc[df[column] == cat1][metric]
    serie2 = df.loc[df[column] == cat2][metric]

    cnt_1 = serie1.shape[0]
    cnt_2 = serie2.shape[0]
    succ_1 = serie1.loc[serie1==1].shape[0]
    succ_2 = serie2.loc[serie2==1].shape[0]

    # z = stats.proportion.test_proportions_2indep(succ_1, cnt_1, succ_2, cnt_2)
    z, p_value = proportion.proportions_ztest(count=np.array([succ_1, succ_2]), nobs=np.array([cnt_1, cnt_2]), alternative='larger')


    if p_value > 0.05:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, {cat1} and {cat2} distributions are equal')
    else:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, {cat1} and {cat2} distributions are different')

    if return_:
        return cnt_1, succ_1, cnt_2, succ_2

def plot_by_category_count(df, col_category,
    title, x_label, y_label, return_table = False
    ):
    '''
    Function to plot a bar horizontal and show a metric
    by different categories
    '''
    # plot_df = df.loc[df['viewed_rate'] != 0]

    plot_df = df[col_category].\
        value_counts().sort_index(ascending=False) / df.shape[0]


    # plot_df = df.groupby('offer_type')['completed_rate'].mean().\
    #     sort_values(ascending=True)

    plt.figure(dpi=85)
    plot_df.plot.barh()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()

    if return_table:
        return plot_df.sort_index(ascending=True)


