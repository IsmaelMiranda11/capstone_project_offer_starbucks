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
from statsmodels.stats import proportion, weightstats


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

def test_means_in_dataframe(df, column, cat1,cat2, metric, return_=False):
    '''
    Function to assert that two proportions are statistically different
    '''
    serie1 = df.loc[df[column] == cat1][metric].to_numpy()
    serie2 = df.loc[df[column] == cat2][metric].to_numpy()

    mean1 = serie1.mean().round(2)
    mean2 = serie2.mean().round(2)

    t, p_value, deg = weightstats.ttest_ind(x1=serie1, x2=serie2, value=0)  

    
    if p_value > 0.05:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, means of {cat1} ({mean1}) and {cat2} ({mean2}) are equal')
    else:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, means of {cat1} ({mean1}) and {cat2} ({mean2}) are are different')

    if return_:
        return t, p_value, deg 



def plot_by_category_count(df, col_category,
    title, x_label, y_label, return_table = False, ax=False
    ):
    '''
    Function to plot a bar horizontal and show the cont of values
    by different categories
    '''

    plot_df = df[col_category].\
        value_counts().sort_index(ascending=False) / df.shape[0]
    
    if not ax:
        f, ax = plt.subplots()
        f.set_dpi(85)

    plot_df.plot.barh(ax=ax, colormap='Spectral_r')
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if not ax:
        plt.show()

    if return_table:
        return plot_df.sort_index(ascending=True)


def plot_by_category_metric(df, col_category, metric, 
    title, x_label, y_label, return_table = False, agg='mean',
    ax=False
    ):
    '''
    Function to plot a bar horizontal and show a metric
    by different categories
    '''
    plot_df = df.groupby(col_category)[metric].agg('mean').sort_index(ascending=False)

    if not ax:
        f, ax = plt.subplots()
        f.set_dpi(85)

    plot_df.plot.barh(ax=ax)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if not ax:
        plt.show()

    if return_table:
        return plot_df.sort_index(ascending=True)


def plot_grid_categories_metric(df, cols_cats, col_cat_to_grid,col_cat_to_acc, metric, 
    title, x_label, y_label, return_table = False, agg='mean'):

    # Group the df
    # plot_df = df.groupby(['offer_type', 'age_quartile'], as_index=False)['viewed_rate'].mean().sort_index(ascending=True)
    plot_df = df.groupby(cols_cats, as_index=False)[metric].\
        mean().sort_index(ascending=True)

    g = sns.FacetGrid(data=plot_df, col=col_cat_to_grid, sharex=True)
    g.map_dataframe(sns.barplot, y=col_cat_to_acc, x=metric)
    g.fig.subplots_adjust(top=0.85)
    g.fig.suptitle(title)
    g.set_xlabels(x_label)
    g.set_ylabels(y_label)

    fig = plt.gcf()

    fig.set_size_inches(18,4)
    plt.show()

    if return_table:
        return plot_df