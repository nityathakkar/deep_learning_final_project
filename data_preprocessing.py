import numpy as np
import pandas as pd
import heapq
import stringdb
import math
from statistics import variance
from sklearn import preprocessing
#from sklearn import to_categorical
import tensorflow as tf
import torch


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

    # Sort the dict in descending order and take top 1000 largest entries in dict
    top_1000 = heapq.nlargest(1000, variance_dict, key=variance_dict.get)

    # Subset dataframe to be 1000 most variant genes
    df_subset = exp_df.loc[:, top_1000]
    return df_subset

def normalize_adj_matrix(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx
    
# Construct gene adjacency network from the selected genes
def create_adj_matrix(exp_df):  
    # elements in matrix represent the confident score between pairs of genes extracted from the gene–gene interaction database (from StringDB)
    # Normalize weights by row sums
    # Use this to build a weighted graph where nodes are genes and edges represent the connection between genes and the normalized confidence scores are weights of edges
    
    
    genes = list(exp_df.columns.values)
    
    string_ids = stringdb.get_string_ids(genes)
    gene_network = stringdb.get_network(string_ids.queryItem)

    # TO CHECK LATER: WHY DUPLICATES CREATED?
    gene_network = gene_network.drop_duplicates()

    gene_network = gene_network.loc[:, ['preferredName_A', 'preferredName_B', 'score']]
    # print(gene_network)

    col_A = np.unique(gene_network.loc[:,'preferredName_A'])
    col_A_updated = np.intersect1d(col_A, genes)

    col_B = np.unique(gene_network.loc[:,'preferredName_B'])
    col_B_updated = np.intersect1d(col_B, genes)

    col_A_diff = np.setxor1d(col_A, genes)
    # print(col_A_diff)
    col_B_diff = np.setxor1d(col_B, genes)
    # print(col_B_diff)

    gene_network_subset1 = gene_network[(gene_network.preferredName_A.isin(col_A_diff) == False)]
    gene_network_subset = gene_network_subset1[(gene_network_subset1.preferredName_B.isin(col_B_diff) == False)]
    # print(gene_network_subset)


    # first, figure out which genes are in gene_network that aren't in genes
    # only keep rows with genes we have (so if one gene name not in colA or colB remove whole row)
    # create adj matrix
    # subset exp_df to be just genes in adj_matrix

    # genes_adj_matrix = np.unique(np.concatenate([col_A, col_B]))  
    genes_adj_matrix = np.unique(np.concatenate([col_A_updated, col_B_updated]))  

    exp_df_subset = exp_df[genes_adj_matrix]

    adj_matrix = pd.DataFrame(index=genes_adj_matrix, columns=genes_adj_matrix)

    np.fill_diagonal(adj_matrix.values, 0) # Fill the diagonal with 0 (we don't want self loops)
    
    for g1 in col_A_updated:
        
        match_1_df = gene_network_subset.loc[gene_network_subset['preferredName_A'] == g1]

        gene2_col = np.unique(match_1_df.loc[:,'preferredName_B'])
    
        for g2 in gene2_col:    
            if math.isnan(adj_matrix.loc[g1,g2]):

                match_2_df = match_1_df.loc[match_1_df['preferredName_B'] == g2]

                adj_matrix.loc[g1,g2] = float(match_2_df['score'])
                adj_matrix.loc[g2, g1] = float(match_2_df['score'])

    adj_matrix = adj_matrix.fillna(0)
    # adj_matrix = adj_matrix.div(adj_matrix.sum(axis=1), axis=0)
    adj_matrix = normalize_adj_matrix(adj_matrix)

    # print(np.all(adj_matrix.values == adj_matrix.values.T))    

    # print(exp_df)
    # exp_df = exp_df[exp_df.columns.intersection(genes_adj_matrix)]
    # print(exp_df)

    # adj_matrix.to_csv('adj_matrix_new.csv')
    return exp_df_subset, adj_matrix


def encode_labels(labels_array):
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels_array.ravel())
    # print(labels[0])
    # labels = to_categorical(labels)
    
    return labels, len(label_encoder.classes_)
    
def spilt_data(gene_exp, labels_array):

    # Convert gene_exp df into an np array
    gene_exp_values = gene_exp.values

    # Convert gene_exp_values and labels_array to tensors
    gene_tensor = tf.convert_to_tensor(gene_exp_values, dtype=tf.float32)
    labels_tensor = tf.convert_to_tensor(labels_array, dtype=tf.int32)

    # Shuffle the data
    num_cells = gene_exp.shape[0]
    ind = np.arange(0, num_cells)
    ind = tf.convert_to_tensor(ind, dtype=tf.int32)

    tf.random.shuffle(ind)

    train_inputs = tf.gather(gene_tensor, ind)
    train_labels = tf.gather(labels_tensor, ind)
    
    train_data = train_inputs[0: int(0.8*num_cells)]
    val_data = train_inputs[int(0.8*num_cells):int(0.9*num_cells)]
    test_data = train_inputs[int(0.9*num_cells):]

    train_labels = train_labels[0: int(0.8*num_cells)]
    val_labels = train_labels[int(0.8*num_cells):int(0.9*num_cells)]
    test_labels = train_labels[int(0.9*num_cells):]

    return train_data, val_data, test_data, train_labels, val_labels, test_labels
    
def get_data(path_exp, path_labels):
    print("Loading in datasets...\n")
    exp_df, labels_array = load_data(path_exp, path_labels)

    print("Removing all 0 columns...\n")
    no_zero_df = remove_zero_col(exp_df)

    print("Normalzing data...\n")
    normalize_df = normalize(no_zero_df)

    print("Calculating most variant genes...\n")
    var_df = calc_variance(normalize_df)

    print("Creating gene adjacency network...\n")
    exp_df_subset, adj_matrix = create_adj_matrix(var_df)

    print("Encoding labels...\n")
    labels, num_classes = encode_labels(labels_array)


    print("Splitting data into train, validation, and test...\n")
    train_data, val_data, test_data, train_labels, val_labels, test_labels = spilt_data(exp_df_subset, labels)

    train_data_npy = train_data.numpy()
    val_data_npy = val_data.numpy()
    test_data_npy = test_data.numpy()
    train_labels_npy = train_labels.numpy()
    val_labels_npy = val_labels.numpy()
    test_labels_npy = test_labels.numpy()

    np.save("train_data", train_data_npy)
    np.save("val_data", val_data_npy)
    np.save("test_data", test_data_npy)
    np.save("train_labels", train_labels_npy)
    np.save("val_labels", val_labels_npy)
    np.save("test_labels", test_labels_npy)

    adj_matrix.to_csv("adj_matrix.csv")
    print("num classes: ", num_classes)

    return train_data, val_data, test_data, train_labels, val_labels, test_labels, adj_matrix, num_classes