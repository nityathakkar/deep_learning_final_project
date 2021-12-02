import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
import heapq

# Load csv file into pandas dataframes
def load_data(path_exp, path_labels):
    exp_df = pd.read_csv(path_exp, index_col=0)
    labels_df = pd.read_csv(path_labels)
    
    return exp_df, labels_df

# Remove genes that have all 0 values for every cell (ie all 0s in a given column)
def remove_zero_col(exp_df):
    exp_df = exp_df.loc[:, (exp_df != 0).all()]
    return exp_df


# Transform gene expression values into log scale and normalize each dataset by min–max scaling
def normalize(exp_df):
    exp_df = np.log2(exp_df)
   #min-max scaling:
    exp_df = (exp_df-exp_df.min())/(exp_df.max()-exp_df.min())
    return exp_df
   

# Calculating variances of the genes across all the cells
# Sort the variances in descending order
# Choose the top 1000 genes as the input of the classifiers

def calc_variance(exp_df):
    variance_dict = {}

    # loop through all the columns and calculate variance down each column (ie for each gene)
    # gene name map to variance
    # sort the dict in descending order
    # take top 1000 entries in dict
    
    for col_name in exp_df.columns:
        variance_dict[col_name] = exp_df.var()[col_name]

    #sorted_dict = dict(sorted(variance_dict.items(), key=lambda item: item[1], reverse=True))
    # TO DO: get first 1000 items
    # top_1000 = take(1000, sorted_dict.iteritems())
    print("dict size before", len(variance_dict))
    print("taking top 10000")
    top_1000 = heapq.nlargest(2, variance_dict, key=variance_dict.get)
    #keys = top_1000.keys()
    print(type(top_1000))
    df_subset = exp_df.loc[:, top_1000]
    print("dict_size_after", len(top_1000))
    return df_subset

# Construct gene adjacency network from the selected genes
def adj_matrix(exp_df):  
    pass


if __name__ == '__main__':
    path_exp = sys.argv[1]
    path_labels = sys.argv[2]

    print("Loading in datasets...")
    exp_df, labels_df = load_data(path_exp, path_labels)
    print(exp_df)

    print("Removing all 0 columns...")
    no_zero_df = remove_zero_col(exp_df)
    print(no_zero_df)

    print("Normalzing data...")
    normalize_df = normalize(no_zero_df)
    print(normalize_df)

    print("Calc var")
    var_df = calc_variance(normalize_df)
    print(var_df)



    
