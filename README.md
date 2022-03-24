# Optimizing Offers Starbucks

#### Content
1. [Installations](#what)
2. [Project Motivation](#why)
3. [File Descriptions](#how)
4. [Results](#results)
5. [Licensing, Authors, and Acknowledgements](#lic)


## Installations <a name="what"></a>

All codes in this repository were written using the python 3.8 version.  
These libraries (with the specific version) are required for the code to run correctly:
- Pandas ( pip install pandas==1.4.1 )  
- Numpy ( pip install numpy==1.22.3 )  
- Seaborn ( pip install seaborn==0.11.2 )  
- Matplotlib ( pip install matplotlib==3.5.1 )
- Sklearn ( pip install scikit-learn==1.0.2 )  
- StatsModels ( pip install statsmodels==0.13.2 )


## Project Motivation <a name="why"></a>

This is the capstone project of Udacity Data Scientist Nanodegree Program.

Here, I will deal with data of offers for members of the reward program from Starbucks.

The goal is to analyze the data and create a machine learning model to increase the effectiveness of offers, increasing the completed rate and the amount of transactions influenced by the offers.


## File Descriptions<a name="how"></a>

The supplied data are in the `data/` folder, all in json format. 

All the process of answering the goal is performed in 5 files.

These are the files to complete the project:

0. **`Data Explanation.ipynb`**: In this notebook, the data provided to carry out the project are explained. All fields of the three datasets (profile, portifolio and transcript) are specified.   

1. **`Data Preparation.py`**: Data must undergo extensive transformation before being analyzed and modeled. In this module, the three datasets are treated and the output are saved in the `data_treated/` folder.

1. **`Profile Treatment.ipynb`**: This project allows works with user profiling. Some users have empty data, which require a treatment. Other opportunity that the data delivers is the use of non-supervised methods of machine learning to identify groups in the data.
In this notebook is made the treatment of the NaN in the dataset profile and the creation of the user cluster.
1. **`Data Exploration.ipynb`**: After the data are in appropriate format, in this notebook the visual and statistical analysis of the data is done. In its end, a statistical model shows which offer would be the best for each user group.
1. **`Data Modeling.ipynb`**: In this notebook, two classification models are training and implemented to direct the best offers based on user profiles.

In addition to this main files, another two modules help in the project:

- **`mapper_id/Mapper IDs.py`**: In the datesets of profile and portifolio IDs are in hash format. To ease work with datasets, this module applies a function to map the IDs as integer and save a dictionary in the folder.
- **`functions/functions.py`**: Module with various auxiliary functions for project execution. There are three categories within the module: statistics, plotting and modeling. The goal is to facilitate the writing of the codes.

If you want, you can make the repository fork and run the files in sequence to follow the project

## Results <a name="results"></a>

The results of each step are commented within the jupyter notebooks.
A complete explanation of the project can be read in this [blog post](https://medium.com/@ismaelfmiranda/lets-have-a-coffee-data-modeling-of-offers-from-starbucks-4f83af93aa23)
  

## Licensing, Authors, and Acknowledgements<a name="lic"></a>

The simulated data is from Starbucks. Grateful for the availability.     
The code can be used as you like.
