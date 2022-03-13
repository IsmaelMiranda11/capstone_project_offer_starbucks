'''
Module to tranform hexacemial ids to int ids
'''

from matplotlib.font_manager import json_dump
import pandas as pd
# from functions.functions import mapper_ids
import json

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

    return map_

def main():
    # Read data
    portfolio = pd.read_json('data/portfolio.json', orient='records', lines=True)
    profile = pd.read_json('data/profile.json', orient='records', lines=True)

    # Map portifolio
    map_protifolio_dict = mapper_ids(portfolio.id)
    map_profile_dict = mapper_ids(profile.id)

    # Export dict as jsons
    # This allows user a fixed dictionary to every running
    json_dump(map_protifolio_dict, 'portifolio_ids.json')
    json_dump(map_profile_dict, 'profile_ids.json')

if __name__ == "__main__":
    main()