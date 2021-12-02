import numpy as np
import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
import heapq
import stringdb
import math

# Load csv file into pandas dataframes
def load_data(path_exp, path_labels):
    exp_df = pd.read_csv(path_exp, index_col=0)
    labels_df = pd.read_csv(path_labels)
    
    return exp_df, labels_df

# TO DO ONCE HAVE REAL DATA!
# remove unlabeled cells and cells labelled as debris and doublets
# def filter_cells(exp_df, labels_df):
    # pass
    

# Remove genes that have all 0 values for every cell (ie all 0s in a given column)
def remove_zero_col(exp_df):
    exp_df = exp_df.loc[:, (exp_df != 0).any(axis=0)]
    return exp_df


# Transform gene expression values into log scale and normalize each dataset by min–max scaling
def normalize(exp_df):
    log_df = np.log2(exp_df+1)
    # Min-max scaling:
    normalize_df = (log_df-log_df.min())/(log_df.max()-log_df.min())
    return normalize_df
   

# Calculating variances of the genes across all the cells
# Sort the variances in descending order
# Choose the top 1000 genes as the input of the classifiers
def calc_variance(exp_df):
    variance_dict = {}

    for col_name in exp_df.columns:
        variance_dict[col_name] = exp_df.var()[col_name]

    # Sort the dict in descending order and take top 1000 largest entries in dict
    top_1000 = heapq.nlargest(1000, variance_dict, key=variance_dict.get)

    # Subset dataframe to be 1000 most variant genes
    df_subset = exp_df.loc[:, top_1000]
    return df_subset

# Construct gene adjacency network from the selected genes
def create_adj_matrix(exp_df):  
    # elements in matrix represent the confident score between pairs of genes extracted from the gene–gene interaction database (from StringDB)
    # Normalize weights by row sums
    # Use this to build a weighted graph where nodes are genes and edges represent the connection between genes and the normalized confidence scores are weights of edges
    # genes = list(exp_df.columns.values)

    genes = ['TP53', 'BRCA1', 'FANCD1', 'FANCL']
    string_ids = stringdb.get_string_ids(genes)
    # print(string_ids.loc[:, 'taxonName'])
    gene_network = stringdb.get_network(string_ids.queryItem)

    # TO CHECK LATER: WHY DUPLICATES CREATED?
    gene_network = gene_network.drop_duplicates()

    gene_network = gene_network.loc[:, ['preferredName_A', 'preferredName_B', 'score']]
    print(gene_network)

    gene_network = np.asarray(gene_network)

    for i in range(len(gene_network)):
        if gene_network[i, 0] > gene_network[i, 1]:
            gene_network[i, 0], gene_network[i, 1] = gene_network[i, 1], gene_network[i, 0]

    gene_network_df = pd.DataFrame(gene_network, index=None, columns=['preferredName_A', 'preferredName_B', 'score'])

    print(gene_network_df)
    col_A = np.unique(gene_network_df.loc[:,'preferredName_A'])
    col_B = np.unique(gene_network_df.loc[:,'preferredName_B'])

    genes = np.unique(np.concatenate([col_A, col_B]))  
    adj_matrix = pd.DataFrame(index=genes, columns=genes)

    np.fill_diagonal(adj_matrix.values, 0) # Fill the diagonal with infinity
    for g1 in col_A:
        match_1_df = gene_network_df.loc[gene_network_df['preferredName_A'] == g1]
        for g2 in col_B:
            if math.isnan(adj_matrix.loc[g1,g2]):
                match_2_df = match_1_df.loc[match_1_df['preferredName_B'] == g2]
                print(g1, g2)
                print(len(match_2_df['score']))
                adj_matrix.loc[g1,g2] = float(match_2_df['score'])
                adj_matrix.loc[g2, g1] = float(match_2_df['score'])

    print(adj_matrix)

    pass


if __name__ == '__main__':
    path_exp = sys.argv[1]
    path_labels = sys.argv[2]

    print("Loading in datasets...\n")
    exp_df, labels_df = load_data(path_exp, path_labels)
    # print(exp_df)

    print("Removing all 0 columns...\n")
    no_zero_df = remove_zero_col(exp_df)
    # print(no_zero_df)

    print("Normalzing data...\n")
    normalize_df = normalize(no_zero_df)
    # print(normalize_df)

    print("Calculating most variant genes...\n")
    var_df = calc_variance(normalize_df)
    # print(var_df)

    print("Creating gene adjacency network")
    adj_matrix = create_adj_matrix(var_df)




    
