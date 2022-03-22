'''
Auxiliary functions to the project

São quatro seções: 
Utilities, Statistics, Plotting and Modeling

Warning!!!
The version of the StatsModels package must be updated to '0.13.2' for this
module run correctly

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
from itertools import combinations


from sklearn.metrics import classification_report, ConfusionMatrixDisplay,confusion_matrix

# Utilities
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

# Statistics

def create_pairs_cat(df, column):
    '''
    Fucntion to create pairs of categories inside a column in dataframe.
    For example:
        Input -> Categories = [A, B, C]
        Output -> [(A,B), (B,C), (A,C)]
    '''
    cats = np.sort(np.array(df[column].unique()))
    combinations_ = combinations(cats, 2)
    return combinations_

def test_proportions_in_dataframe(df, column, cat1,cat2, metric, return_=False, return_2 = False):
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
        result = 'equal'
    else:
        print(f'p-value of {p_value.round(3)}. With confidence value of 0.05, {cat1} and {cat2} distributions are different')
        result = 'different'

    # Return statistics
    if return_:
        return cnt_1, succ_1, cnt_2, succ_2
    
def test_means_in_dataframe(df, column, cat1,cat2, metric, print_result=True, return_=False,
    return_2 = False):
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
        result = 'equal'
    else:
        result = 'different'

    if print_result:
        if result == 'equal':
            print(f'(==) Means of {cat1} ({mean1}) and {cat2} ({mean2}) are equal (diff = {(mean1-mean2).round(2)}). p-value = {p_value.round(3)}, confidence value = 0.05')
        else:
            print(f'(!=) Means of {cat1} ({mean1}) and {cat2} ({mean2}) are different (diff = {(mean1-mean2).round(2)}). p-value = {p_value.round(3)}, confidence value = 0.05')
            
            
    # Return statistics
    if return_:
        return mean1, mean2

    if return_2:
        return result, mean1-mean2

def best_groups_means(df, column, metric):
    '''
    Function to create a dataframe with best groups based in mean of a metric.
    '''
    # Group dataframe
    df_group = df.groupby(column)[metric].mean().sort_values(ascending=False)
    categories = np.array(df_group.index)
    values = np.array(df_group.values)

    best_groups = []
    gains = []
    
    max_group = categories[0]
    best_value = values[0].round(2)
    best_groups.append(max_group)
    
    for group in categories[1:]:
        compare, diff = test_means_in_dataframe(df,column, metric=metric, cat1=max_group, cat2=group, return_2=True, print_result=False)
        if compare == 'equal':
            best_groups.append(group)
        if compare == 'different':
            gains.append(diff)
    
    if gains:
        min_gain = np.array(gains).min()
    else:
        min_gain = 0
        
    df = pd.DataFrame({'best_groups':[best_groups], 'best_value':best_value,
        'min_gain':min_gain}, index=[column])
    return df

def resume_differences(df, category):
    '''
    TODO docstring
    '''

    # Case it is informational, do not print complete informations
    if df['offer_type'].unique()[0] != 'informational':
        print('\n---Complete Rate---------\n')
        # Complete
        plot_by_category_metric(df=df, 
                col_category=category,
                metric='completed_after_view_rate',
                title='Offer complete rate',
                x_label='Completed rate',
                y_label=category,
                xlims=[0,1]
                )

        groups = create_pairs_cat(df, category)

        for group in groups:
            test_means_in_dataframe(df, column=category, 
                cat1=group[0], cat2=group[1], metric='completed_after_view_rate')
        plt.show()

    print('\n---Transactions---------\n')

    # Transactions
    plot_by_category_metric(df=df, 
            col_category=category,
            metric='tra_offer_infl',
            title='Average transaction',
            x_label='Transaction $',
            y_label=category
            )

    groups = create_pairs_cat(df, category)

    for group in groups:
        test_means_in_dataframe(df, column=category, 
            cat1=group[0], cat2=group[1], metric='tra_offer_infl')
    plt.show()

def resume_best_for_groups(df,category,metric):
    values = df[category].unique()

    dfs = []

    for value in values:
        local_df = df.loc[df[category] == value]
        df_best_groups = best_groups_means(local_df, 'offer_type', metric)
        df_best_groups['category'] = value
        dfs.append(df_best_groups.set_index('category'))

    return pd.concat(dfs).sort_index()

# Plotting
def plot_grid_categories_metric(df, cols_cats, col_cat_to_grid,col_cat_to_acc, metric, 
    title, x_label, y_label, return_table = False, agg='mean'):
    '''
    
    '''

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
    ### Description
        The function receive a dataframe with columns of catergories and 
        metrics. The main output is a horizontal bar graph. 
        If two categories columns are provieded, the function plots the first
        category opened by the second.
        The standard aggregation is 'mean'. If 'proportion' is provieded, then
        function will calculate the percentage of each category over the 
        total of rows in dataframe.
    ### Input
        df - (dataframe) Pandas dataframe with data.   
        col_category - (str or list of str) One category column to group
        data in df. If a list of string is provided, the group operation
        is done with all of it and the plot will be done with two groups.         
        metric - (str) The metric column in df.  
        title - (str) Title of graph.  
        x_label - (str) Title for x axes.  
        y_label - (str) Title for y axes.  
        return_table (bool) If true, return the dataframe grouped that
        is used to plot.  
        agg - (str, {'mean', 'proportion'}) If mean, the group operation
        is done by mean of metric. If 'proportion', the metric column is 
        just a anchor for calculate the proportiton of group in whole dataframe.  
        ax - (bool) If False, the function will create a matplotlib figure. If
        it is provided, use the value provided.  
        xlims - (bool, list) A list of values to limit the x axes.  
        dodge - (bool) If two categories it is provided and dodge is True, the
        graph will separate the categories horizontally.  
    ### Output
        In general, the function plot a graph.  
        If return_table == True:  
            plot_df - (dataframe) Dataframe used to plot the graph  
    ''' 

    # Group data
    plot_df = df.groupby(col_category, as_index=False)[metric] # group

    if agg == 'proportion':
        plot_df = plot_df.agg('count') 
        plot_df = plot_df.assign(proportion=lambda x: x[metric]/x[metric].sum())
        metric='proportion' # changes the metric column, used to plot
    else:
        plot_df = plot_df.agg(agg)

    plot_df = plot_df.sort_index(ascending=False) # sort

    # If there is not a pre created axes, create one with figure
    if not ax:
        f, ax = plt.subplots()
        f.set_dpi(85)

    # If the col_category is a list, plot with hue parameter of seaborn.
    # First category will be the y column and second will be the legend.
    if isinstance(col_category, list):
        g = sns.barplot(orient='h', data=plot_df, x=metric, y=col_category[0], 
            hue=col_category[1], dodge=dodge, palette='tab10', ax=ax)
        sns.move_legend(g, "lower center", bbox_to_anchor=(.5, 1), 
            ncol=len(col_category[1]), title=None, frameon=False)
        pad_=34 # set pad for figure title
    else:
        g = sns.barplot(orient='h', data=plot_df, x=metric, y=col_category, 
        ax=ax)
        pad_=None # set pad for figure title
    
    # Graph layout
    ax.set_title(title, pad=pad_, loc='left')
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    if xlims:
        ax.set_xlim(xlims)

    if not ax:
        plt.show()

    if return_table:
        return plot_df.sort_index(ascending=True)

def plot_grid_metrics(df, col_category, dodge=False):
    '''
    Function to plot a grid. This function uses the plot_by_category_metric to 
    plot different graph inside a subplots objet of matplotlib.
    Input:
        df - (dataframe wit data)
        col_category - (str or list of str) Column with category to plot
        graphs or a list of categories.
        dodge - (bool) If two categories it is provided and dodge is True, the
        graph will separate the categories horizontally.  
    Output:
        Plot grid graph
    '''

    # Treat the name of main category
    if isinstance(col_category, list):
        cat_name = col_category[0].replace('_',' ').title()
        top_adjust = .7 # no legends
    else:
        cat_name = col_category.replace('_',' ').title()
        top_adjust = .8 # space for legends
        
    # Create a subplot figure 
    f, axs = plt.subplots(nrows=1, ncols=3, sharey=True, 
        gridspec_kw={'wspace':0.08}
        )
    f.subplots_adjust(top=top_adjust)
    f.set_size_inches(20,5)
    f.set_dpi(100)
    
    f.suptitle('Metrics by ' + cat_name)

    # Metric: Visualization Rate
    plot_by_category_metric(df=df, 
        col_category=col_category,
        metric='valid_view_rate',
        title='Offer viewed rate',
        x_label='Viewed rate',
        y_label=cat_name,
        xlims=[0,1],
        ax=axs[0],
        dodge = dodge
        )

    plot_df = df.loc[df['valid_view_rate'] != 0] # just 

    # Metric: Complete Rate
    plot_by_category_metric(df=plot_df, 
        col_category=col_category,
        metric='completed_after_view_rate',
        title='Offer complete rate',
        x_label='Complete rate',
        y_label=cat_name,
        xlims=[0,1],
        ax=axs[1],
        dodge = dodge
        )

    # Metric: Transactions
    plot_by_category_metric(df=df, 
        col_category=col_category,
        metric='tra_offer_infl',
        title='Average transaction',
        x_label='Transactions $',
        y_label=cat_name,
        ax=axs[2],
        dodge = dodge
        )


# Modeling

def evaluate_model(model, X_train, y_train, X_test, y_test):
    '''
    TODO 
    '''
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the perfomance
    print('Perfomance with Train\n')
    print(classification_report(y_train, y_pred_train))
    ConfusionMatrixDisplay(confusion_matrix(y_train, y_pred_train)).plot()
    plt.show()

    print('Perfomance with Test\n')
    print(classification_report(y_test, y_pred_test))
    ConfusionMatrixDisplay(confusion_matrix(y_test, y_pred_test)).plot()
    plt.show()

def expand_dataframe(df1, df2):
    df1['key'] = 0
    df2['key'] = 0

    df = df1.merge(df2, on='key').drop(columns='key')
    


    return df

def evaluate_model_reg(model, X_train, y_train, X_test, y_test):
    '''
    TODO 
    '''
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Evaluate the perfomance
    print('Perfomance with Train\n')
    print(f'R2: {model.score(X_train, y_train)}')

    print('Perfomance with Test\n')
    print(f'R2: {model.score(X_test, y_test)}')

    print('Residuals plot')
    plt.scatter(y_pred_train, abs(y_pred_train-y_train))
    plt.scatter(y_pred_test, abs(y_pred_test-y_test))
    plt.title('\nResidual plot')
    plt.show()

def expand_dataframe(df1, df2):
    df1['key'] = 0
    df2['key'] = 0

    df = df1.merge(df2, on='key').drop(columns='key')
    


    return df