import numpy as np
import pandas as pd
import tensorflow as tf

# encoder part of the GCN model consists of a GCN layer followed by a maxpooling layer, 
# a flatten layer and a fully connected (FC) layer. While the decoder part consists of a 
# FC layer to reconstruct the gene expression values.

class GCN():

def gcn_layer(adj_matrix, gene_exp):
    adj_binary = pd.DataFrame(np.where(adj_matrix != 0, 1, 0), index=adj_matrix.index, columns=adj_matrix.columns)
    num_edges = adj_binary.apply(sum(), axis=1)
    
    
    adj_array = adj_matrix.to_numpy()
    gene_array = gene_exp.to_numpy()
    gene_exp_T = tf.transpose(gene_array)

    N = adj_array.shape[0]

    # for each row, find number of non-zero values
    # convert adj matrix where if 0 --> 0, if not zero --> 1
    # calcualte row sum
    D = np.zeros((N,N))
    np.fill_diagonal(D, num_edges)

    I = np.identity(N)

    L = np.subtract(D,adj_array)

    L = I + np.matmul(np.pow(D, -1/2), np.matmul(adj_array, np.pow(D, 1/2)))

    eig_vec, eig_val, eig_vec_T = np.linalg.svd(L)


    # cheby blah

    # max pool

    # flatten

    # output layer is a vector of size 32

    # MSE loss
    

def decoder():
    # dense layer and relu

    # input and output is num of genes (N)
    pass

def NN_gene_exp():
    # two dense layers + relu (size = 256  and 32)
    pass

    
def final_layer():
    # concatenate outputs from encoder and NN_gene_exp
    # softmax probabilites     
