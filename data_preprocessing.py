import numpy as np
import pandas as pd
import heapq
import stringdb
import math
from statistics import variance
from sklearn import preprocessing
from sklearn import to_categorical
import tensorflow as tf


# Load csv file into pandas dataframes
def load_data(path_exp, path_labels):
    exp_df = pd.read_csv(path_exp, index_col=0)
    labels_df = pd.read_csv(path_labels)
    labels_array = labels_df.values
    return exp_df, labels_array

# TO DO ONCE HAVE REAL DATA!
# remove unlabeled cells and cells labelled as debris and doublets
# def filter_cells(exp_df, labels_df):
    # assert len exp_df and len labels_df are the same
    # if any of labels_df is debris or doublets, record index and remove
    # use recorded index to remove those rows from exp_df
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

    var_values = exp_df.apply(variance, axis=0)
    keys = exp_df.columns
    variance_dict = dict(zip(keys, var_values))

    # for col_name in tqdm(exp_df.columns):
    #     variance_dict[col_name] = exp_df.var()[col_name]

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
    
    
    genes = list(exp_df.columns.values)
    # print(genes)
    
    string_ids = stringdb.get_string_ids(genes)
    gene_network = stringdb.get_network(string_ids.queryItem)
    # gene_network.to_csv('network.csv')

    # TO CHECK LATER: WHY DUPLICATES CREATED?
    gene_network = gene_network.drop_duplicates()

    gene_network = gene_network.loc[:, ['preferredName_A', 'preferredName_B', 'score']]
    # print(gene_network)

    col_A = np.unique(gene_network.loc[:,'preferredName_A'])
    col_B = np.unique(gene_network.loc[:,'preferredName_B'])

    genes = np.unique(np.concatenate([col_A, col_B]))  
    adj_matrix = pd.DataFrame(index=genes, columns=genes)
    # count = 0
    np.fill_diagonal(adj_matrix.values, 0) # Fill the diagonal with 0 (we don't want self loops)
    for g1 in col_A:
        match_1_df = gene_network.loc[gene_network['preferredName_A'] == g1]
        # print(g1)
        # print(match_1_df)
        gene2_col = np.unique(match_1_df.loc[:,'preferredName_B'])
        for g2 in gene2_col:
            if math.isnan(adj_matrix.loc[g1,g2]):
                # count +=1
                # print(g2)
                match_2_df = match_1_df.loc[match_1_df['preferredName_B'] == g2]
                # print(match_2_df)
                # print(match_2_df['score'])
                adj_matrix.loc[g1,g2] = float(match_2_df['score'])
                adj_matrix.loc[g2, g1] = float(match_2_df['score'])

    adj_matrix = adj_matrix.fillna(0)
    adj_matrix = adj_matrix.div(adj_matrix.sum(axis=1), axis=0)
    # print("filled #: ", count)
    return adj_matrix


def encode_labels(labels_array):
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels_array)
    print(labels[0])
    # labels = to_categorical(labels)
    
    return labels, len(label_encoder.classes_)
    
def spilt_data(gene_exp, labels_array):

    # Shuffle the data
    num_cells = gene_exp.shape[0]
    ind = np.arange(0, num_cells)
    tf.random.shuffle(ind)

    train_inputs = tf.gather(gene_exp, ind)
    train_labels = tf.gather(labels_array, ind)
    
    
    train_ind = ind[0: int(0.8*num_cells)]
    val_ind = ind[int(0.8*num_cells):int(0.9*num_cells)]
    test_ind = ind[int(0.9*num_cells):]
    
    train_data = train_inputs[train_ind]
    val_data = train_inputs[val_ind]
    test_data = train_inputs[test_ind]

    train_labels = train_labels[train_ind]
    val_labels = train_labels[val_ind]
    test_labels = train_labels[test_ind]

    return train_data, val_data, test_data, train_labels, val_labels, test_labels
    
def get_data(path_exp, path_labels):
    print("Loading in datasets...\n")
    exp_df, labels_array = load_data(path_exp, path_labels)
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
    # to_save = var_df.T
    # to_save.to_csv('gene_exp.csv')

    print("Creating gene adjacency network...\n")
    adj_matrix = create_adj_matrix(var_df)
    print(adj_matrix)

    print("Encode labels...\n")
    labels, num_classes = encode_labels(labels_array)

    print("Splitting data into train, validation, and test...\n")
    train_data, val_data, test_data, train_labels, val_labels, test_labels = spilt_data(var_df, labels)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels, adj_matrix, num_classes