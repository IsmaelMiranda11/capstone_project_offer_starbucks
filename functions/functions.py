'''
Auxiliary module to the project
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
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
    Function to assert that two proportions are statistically different.
    Description
        In a dataframe with a column with values of 1 or 0, test if the 
        proportion of 1s between two categories are statistically different.
        The null hypothesis is p(cat1) = p(cat2) and alternative is 
        p(cat1) != p(cat2). The decision rule is p-value > 0.05, accept
        the null hypothesis.
    Input
        df - (dataframe) Pandas dataframe with data
        column - (str) Column of df to filter the categories
        cat1 - (str) Category 1 to filter in dataframe
        cat2 - (str) Category 2 to filter in dataframe
        metric - (str) Column with 1s and 0s in dataframe
        return_ - (bool) If true, return statistics calculated
    Output
        If return_ == True:
            cnt_1 - (int) Count of obeservations for category 1
            succ_1 - (int) Count of obeservations of 1s for category 1
            cnt_2 - (int) Count of obeservations for category 2
            succ_2 - (int) Count of obeservations of 1s for category 2
    '''

    # Filter the dataframe for two categories
    serie1 = df.loc[df[column] == cat1][metric]
    serie2 = df.loc[df[column] == cat2][metric]

    # Calculate the statistics to use in teste
    cnt_1 = serie1.shape[0]
    cnt_2 = serie2.shape[0]
    succ_1 = serie1.loc[serie1==1].shape[0]
    succ_2 = serie2.loc[serie2==1].shape[0]

    # Test for two independent samples
    z, p_value = proportion.proportions_ztest(count=np.array([succ_1, succ_2]),
        nobs=np.array([cnt_1, cnt_2]), 
        alternative='larger')

    # Evaluation of p-value
    if p_value > 0.05:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, {cat1} and {cat2} distributions are equal')
    else:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, {cat1} and {cat2} distributions are different')

    # Return statistics
    if return_:
        return cnt_1, succ_1, cnt_2, succ_2


def test_means_in_dataframe(df, column, cat1,cat2, metric, return_=False):
    '''
    Function to assert that two means are statistically different.
    Description
        In a dataframe with numeric column test if the mean for two 
        categories are statistically different. The null hypothesis is 
        m(cat1) = m(cat2) and alternative is m(cat1) != m(cat2). The 
        decision rule is p-value > 0.05, accept the null hypothesis.
    Input
        df - (dataframe) Pandas dataframe with data
        column - (str) Column of df to filter the categories
        cat1 - (str) Category 1 to filter in dataframe
        cat2 - (str) Category 2 to filter in dataframe
        metric - (str) Column with 1s and 0s in dataframe
        return_ - (bool) If true, return statistics calculated
    Output
        If return_ == True:
            mean1 - (float) Mean for category 1
            mean2 - (float) Mean for category 2
    '''

    # Filter the dataframe for two categories
    serie1 = df.loc[df[column] == cat1][metric].to_numpy()
    serie2 = df.loc[df[column] == cat2][metric].to_numpy()
    
    # Calculate the statistics to use in teste
    mean1 = serie1.mean().round(2)
    mean2 = serie2.mean().round(2)
    
    # Test mean difference for two independent samples
    _, p_value, _ = weightstats.ttest_ind(x1=serie1, x2=serie2, value=0)  

    # Evaluation of p-value
    if p_value > 0.05:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, means of {cat1} ({mean1}) and {cat2} ({mean2}) are equal')
    else:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, means of {cat1} ({mean1}) and {cat2} ({mean2}) are are different')

    # Return statistics
    if return_:
        return mean1, mean2


def plot_by_category_count(df, col_category,
    title, x_label, y_label, return_table = False, ax=False, xlims=False
    ):
    '''
    Function to plot a bar horizontal and show the cont of values
    by different categories    
    '''

    # Group data
    plot_df = df[col_category].value_counts().sort_index(ascending=False) / df.shape[0]
    
    if not ax:
        f, ax = plt.subplots()
        f.set_dpi(85)

    # Colors
    size = plot_df.shape[0]
    colors = ['lightseagreen', 'teal', 'steelblue']

    plot_df.plot.barh(ax=ax, color=colors)
    # sns.barplot(plot_df)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    
    if xlims:
        ax.set_xlim(xlims)

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


def plot_by_category_metric(df, col_category, metric, title, x_label, y_label, 
    return_table = False, agg='mean', ax=False, xlims=False, dodge=False):
    '''
    Function to plot a bar horizontal and show a metric by different categories.
    Description
        The function receive a dataframe with columns of caterogires and 
        metrics. The main output is a horizontal bar graph. 
        If two categories columns are provieded, the function plots the first
        category divided by second.
        The standard aggregation is 'mean'. If 'proportion' is provieded, then
        function will calculate the percentage of each categorie over the 
        whole dataframe.
     Input
        df - (dataframe) Pandas dataframe with data
        col_category - (str or list of str) One category column to group
        data in df. If a list of string is provided, the group operation
        is done with all of it. 

    '''

    # Group data
    if agg == 'mean':
        plot_df = df.groupby(col_category, as_index=False)[metric].\
            agg(agg).sort_index(ascending=False)
    elif agg == 'proportion':
        plot_df = df.groupby(col_category, as_index=False)[metric].\
            agg('count').assign(proportion=lambda x: x[metric]/x[metric].sum())
        metric='proportion' # change of metric column

    # If there is not a pre created axes, create one with figure
    if not ax:
        f, ax = plt.subplots()
        f.set_dpi(85)

    # If the col_category is a list, plot with hue
    # parameter of seaborn
    if isinstance(col_category, list):
        # Plot the graph
        g = sns.barplot(orient='h', data=plot_df, x=metric, y=col_category[0], 
            hue=col_category[1], dodge=dodge, palette='tab10',ax=ax)
        sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), 
            ncol=len(col_category[1]), title=None, frameon=False)
        pad_=34 # set pad for figure title
    else:
        g = sns.barplot(orient='h', data=plot_df, x=metric, y=col_category, 
        ax=ax)
        pad_=None # set pad for figure title
    
    ax.set_title(title, pad=pad_, loc='left')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if xlims:
        ax.set_xlim(xlims)

    if not ax:
        plt.show()

    if return_table:
        return plot_df.sort_index(ascending=True)