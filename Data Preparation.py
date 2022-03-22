
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import seaborn as sns
import os
import sys
import importlib
import functions as aux_fun
importlib.reload(aux_fun)
from datetime import datetime as dti

# Reading data
portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
profile = pd.read_json('data/profile.json', orient='records', lines=True)
transcript = pd.read_json('data/transcript.json', orient='records', lines=True)

# Mapping to integer dictionaries
map_portifolio = json.load(open('mapper_id/portifolio_ids.json'))
map_profile = json.load(open('mapper_id/profile_ids.json'))

# Functions
def get_total_transaction(user):
    '''
    Some transactions might not be related with offer received, but just
    a commom one. This function get the total of transactions made by
    an user, because of an offer or not
    Input:
        user - the user id
    Output:
        total_tra - (float) total transaction
    '''

    cond = transcript.person == user #1510
    user_df = transcript.loc[cond]

    # Transactions
    transaction_df = get_subset(user_df, 'transaction', '_tra', ['amount'])

    total_tra = transaction_df['amount_tra'].sum()

    return total_tra

def get_subset(user_df, type, suffix, dict_keys):
    '''
    Get a dataset with different events and separates a subsets
    with specific type
    Input:
        user_df - (dataframe) - datafram with event of an user
        type - (string) - 'offer received', 'offer viewed', 'offer completed', 
        'transaction'
        suffix - (string) - a suffix to indtify the variables
        dict_keys - (list) - list of dict keys to extract from value column
    Output:
        df - (dataframe) - dataframe with subset
    '''
    df = user_df.loc[user_df.event==type]
    df = df[['time', 'value']]
    df = df.rename(columns={'time': 'time' + suffix})
    
    if type == 'offer received':
        df['time_next'] = df.shift(-1, 
            fill_value=transcript.time.max())['time' + suffix]
    
    # Extract value from dict
    for key in dict_keys:
        key_ = key.replace(' ', '_') # to replace empty space
        df[key_ + suffix] = df['value'].map(lambda d: d.get(key))
    df = df.drop(columns='value')

    return df 

def merge_and_filter(df_left, 
    df_right, on_left, on_right,
    col_filter, col_compare):
    '''
    Merge two dataframes and apply a filter in result
    Input:
        df_left - (dataframe) - left dataframe to merge
        df_right - (dataframe) - right dataframe to merge
        on_left - (string) - key to use in merge for left
        on_right - (string) - key to use in merge for right
        col_filter - (string) - column to apply some filters
        col_compare - (list of string) - list with columns to apply filters
    Output:
        df - (dataframe) - dataframe merged and filtered 
    '''

    df = df_left.merge(df_right, 
    left_on=on_left,
    right_on=on_right,
    how='left'
    )
    
    condition1 = (df[col_filter] >= df[col_compare[0]]) 
    # Se <, tira alguns valores, se <= fica mais. Testar os dois
    condition2 = (df[col_filter] <= df[col_compare[1]])

    condition3 = (pd.isna(df[col_filter])) # keep not seen offers in datase
    
    df = df.loc[condition1 & condition2 | condition3]

    return df

def guarantee_viewed(original, merged):
    '''
    The main point of offer user table is that it must 
    have all offer received by an user.
    Some merge operations can vanish rows of viewed offers.
    This function check original received offers and assemble
    a complete dataset
    Input:
        original - (dataframe) Dataframe with all received offers
        merged - (dataframe) Dataframe result of a merge that could
        vanish a offer received row.
    Output:
        all_df - (dataframe) A complete dataframe with all received offer
    '''
    original = original.set_index(['time_rec', 'offer_id_rec'])
    merged_index = merged.set_index(['time_rec', 'offer_id_rec'])
    
    index_original = original.index
    index_merged = merged_index.index 
    
    # Test indexs. What index is in original but not in merged
    index_exception = ~(index_original.isin(index_merged))
    # Filter the original with missing index
    original_miss = original.loc[index_exception].reset_index()
    # Join tables
    all_df = pd.concat([merged, original_miss]).reset_index(drop=True)

    return all_df

def validate_view(row):
    '''
    Auxiliar function to account for valid views
    Input: row, a dataframe coming from apply lambda
    Outpu: the validation of viewed offer
    '''
    if not pd.isna(row['time_vie']):
        if row['time_vie'] <= row['period_max']:
            return 1 
        else:
            return 0
    else:
        return np.nan

def validate_complete(row):
    '''
    Auxliar function to account the offer complete after viewd
    Input: row, a datafram row coming from apply lambda
    Outpu: complete offer after viewed
    '''
    if not pd.isna(row['time_com']):
        if row['time_com'] >= row['time_vie']: 
            return 1 
        else:
            return 0
    else:
        return np.nan

def get_offer_table_user(user):
    '''
    Main function to extract informations about offers made for users.
    The ideia is analyse timeline of received, viewed and completed offer
    made by user and build a dataset with this consolitaded information.
    Input:
        user - the user id
    Output:
        offer_df - dataframe with consolitaded informations
    '''
    cond = transcript.person == user #1510
    user_df = transcript.loc[cond]

    # Resume do protifolio
    short_portifolio = portfolio[['id', 'duration']]

    # 1 - Creating the subsets
    # Received offers
    received_df = get_subset(user_df,'offer received', '_rec', ['offer id'] )
    # If user did not received any offer, skip it and return empty dataframe
    if received_df.size == 0:
        return pd.DataFrame()

    # Get info from portifolio
    received_df = received_df. \
        merge(short_portifolio, left_on='offer_id_rec', right_on='id'). \
        drop(columns='id')
    # Viewed offers
    viewed_df = get_subset(user_df, 'offer viewed', '_vie', ['offer id'] )
    # Completed offers
    completed_df = get_subset(user_df, 'offer completed', '_com', ['offer_id', 'reward'])
    # Transactions
    transaction_df = get_subset(user_df, 'transaction', '_tra', ['amount'])

    # 2 - Analizing the timeline
    # 2.1 - Visualized
    # Get the visualized offers
    offer_df = merge_and_filter(received_df, viewed_df, 
        'offer_id_rec', 'offer_id_vie',
        'time_vie', ['time_rec', 'time_next']
        )
    # offer_df = offer_df.sort_values('time_rec')

    # Calculating the max time valid for offer
    offer_df['period_max'] = offer_df.time_rec + offer_df.duration*24

    offer_df['valid_view'] = offer_df.apply(lambda row: validate_view(row), axis=1)

    offer_df['time_vie_next'] = offer_df.shift(-1, 
            fill_value=transcript.time.max())['time_vie'].bfill()


    # 2.2 - Complete offers
    # Complete offers
    offer_rec_vie = offer_df.copy() # get a copy before merge
    offer_df = merge_and_filter(offer_df, completed_df,
        'offer_id_rec', 'offer_id_com',
        'time_com', ['time_rec', 'period_max']
        )
    # offer_df = offer_df.sort_values('time_rec')

    # The same offer can be sent to a user and be completed together, 
    # generating duplicates
    offer_df = offer_df.drop_duplicates(
        subset=['time_rec', 'offer_id_rec', 'time_vie']
        )

    offer_df['completed_after_view'] = offer_df.apply(lambda row: validate_complete(row), axis=1)

    offer_df = guarantee_viewed(offer_rec_vie, offer_df) # all offers in dataset

    # 2.3 - Transactions
    # Iterate over offers dataset and searching in the transactions the intervals
    # considered to be influenced by an offer

    offer_df['tra_offer_infl'] = 0.0000

    for idx, _ in offer_df.iterrows():
        time_vie  = offer_df['time_vie'].at[idx]
        time_max  = offer_df['period_max'].at[idx]
        time_vie_next = offer_df['time_vie_next'].at[idx]
        time_com = offer_df['time_com'].at[idx]
        
        # Check if time_vie is na. If so, there is no transaction for it
        if pd.isna(time_vie):
            offer_df['tra_offer_infl'].at[idx] = np.NaN
            continue

        # Case not complete offer, trate this as maximum value
        if pd.isna(time_com):
            time_com = transcript.time.max()

        # Initialize variable
        sum_tra_infl = 0
        # Itarete over transactions
        for jdx, _ in transaction_df.iterrows():
            # Time of transactions
            time_tra = transaction_df['time_tra'].at[jdx]
            amo_tra  = transaction_df['amount_tra'].at[jdx]
        
            if (time_tra >= time_vie) and \
                (time_tra < time_vie_next and time_tra <= time_max and \
                time_tra <= time_com):
                sum_tra_infl += amo_tra
        
        # Assing to that offer
        offer_df['tra_offer_infl'].at[idx] = sum_tra_infl
    

    # 2.4 - Final
    # Getting the status for offers
    offer_df['viewed'] = offer_df.apply(lambda r: 
        1 if not pd.isna(r['time_vie']) else 0,
        axis=1
        )

    offer_df['completed'] = offer_df.apply(lambda r: 
        1 if not pd.isna(r['time_com']) else 0,
        axis=1
        )

    # Selecting just necessary columns
    offer_df = offer_df[['offer_id_rec', 'valid_view', 'tra_offer_infl',
       'reward_com', 'completed_after_view', 'viewed', 'completed']]

    return offer_df

def group_offer_df(offer_df, map_dict=map_portifolio):
    '''
    After an offer dataframe be created for an user, this function
    group the data and get dummies for each offer present in dataset
    Input:
        offer_df - (dataframe) datafram with data about offer for a user
        map_dict - (dict) a dictonary create initialy to converte hex to int
    Output:
        offer_df - (dataframe) dataframe with dummies variables and grouped
        data

    About metrics
        viewed_rate - the rate of views for an offer
        completed_rate - the rate of completes for an offer
        tra_offer_infl - the total of transaction because of an offer
        valid_view - for viewed offers, the rate of vizualizations in 
        validy period of an offer
        completed_after_view_rate - for complete offer, the rate of that
        was complete after was visualize
        reward - for complete offer, the total of reward won by user
    '''
    # Check if the offer_df is not empty
    if offer_df.size == 0:
        return pd.DataFrame()
    
    # Group the data and create the metrics
    offer_df = offer_df.groupby('offer_id_rec', as_index=False).agg(
        viewed_rate=('viewed', 'mean'),
        completed_rate=('completed', 'mean'),
        tra_offer_infl=('tra_offer_infl', 'sum'),
        valid_view_rate=('valid_view', 'mean'),
        completed_after_view_rate=('completed_after_view', 'mean'),
        reward_won=('reward_com', 'sum')
        )


    offer_df['offer_id_rec'] = offer_df['offer_id_rec'].map(map_dict)

    return offer_df

def create_user_offer_df(user):
    '''
    Function to get the user and apply previous functions
    to extract informations about offers
    Input:
        user - (str, int) user id
    Output:
        user_offer_df - (dataframe) dataframe with informations about user and
        offers
    '''
    user_offer_df = get_offer_table_user(user)
    user_offer_df = group_offer_df(user_offer_df)
    user_offer_df['user_id'] = user

    user_offer_df['user_id'] = user_offer_df['user_id'].map(map_profile)

    return user_offer_df

def generate_user_offer():
    # Iterate by user and get dataframes
    users = transcript.person.unique()
    # np.random.shuffle(users)

    dfs = []
    cnt = 0
    for user in users:
        cnt += 1
        df = create_user_offer_df(user)
        dfs.append(df)
        print((cnt/len(users))*100, end="\r")

    user_offer_df = pd.concat(dfs).reset_index(drop=True)
    user_offer_df.to_csv('user_offer.csv', index=False)

def generate_user_transactions():
    '''
    Function to extract the total of transaction made 
    by an user
    '''
    tra_user_df = transcript[['person']].drop_duplicates().reset_index(drop=True)
    tra_user_df['trans_amount'] = tra_user_df['person'].apply(get_total_transaction)
    tra_user_df['person'] = tra_user_df['person'].map(map_profile)
    tra_user_df.to_csv('user_transactions.csv', index=False)

# def mapper_hex_id(profile, portfolio):
#     '''
#     Function to map hexadecimal ids to interger ids. Create 
#     a new column in existing datasets
#     '''
#     profile['user_id'] = profile.id.map(map_profile)
#     portfolio['offer_id'] = portfolio.id.map(map_portifolio)

#     return profile, portfolio

def generate_datasets():
    '''
    Function to extract the datasets of profile and portfolio
    to be used in others analysis
    '''
    # To map
    profile['user_id'] = profile.id.map(map_profile)
    portfolio['offer_id'] = portfolio.id.map(map_portifolio)
    # Save as csv files
    profile.to_csv('profile.csv', index=False)
    portfolio.to_csv('portfolio.csv', index=False)



generate_user_offer()
generate_user_transactions()
generate_datasets()
