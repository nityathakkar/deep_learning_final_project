import numpy as np
import pandas as pd
import scipy.sparse as sp
import argparse
import sys

def build_adj_weight(idx_features, net_filepath):
    """
    @idx_features: pandas dataframe of [gene x cell], df.index should be gene offcial name.
    @net_filepath: the path of the gene-gene interaction network
    """
    idx_features = pd.read_csv(idx_features, index_col=0)
    edges_unordered =  pd.read_csv(net_filepath, index_col = None, usecols=['preferredName_A','preferredName_B','score']) 
#    edges_unordered = np.asarray(edges_unordered[['protein1','protein2','combined_score']])   # Upper case.
    edges_unordered = edges_unordered.drop_duplicates()
    edges_unordered = np.asarray(edges_unordered) 
    
    
    idx = []
    mapped_index = idx_features.index.str.upper() # if data.index is lower case. Usoskin data is upper case, do not need it.
    print(mapped_index)
    for i in range(len(edges_unordered)):
        print(edges_unordered[i,0], edges_unordered[i,1])
        if edges_unordered[i,0] in mapped_index and edges_unordered[i,1] in mapped_index:
            idx.append(i)
    edges_unordered = edges_unordered[idx]
    print ('idx_num:',len(idx))
    del i,idx
    
    # build graph
    idx = np.array(mapped_index)
    idx_map = {j: i for i, j in enumerate(idx)} # eg: {'TSPAN12': 0, 'TSHZ1': 1}
    # the key (names) in edges_unordered --> the index (which row) in matrix
    print(idx_map)
    edges = np.array(list(map(idx_map.get, edges_unordered[:,0:2].flatten())),
                     dtype=np.int32).reshape(edges_unordered[:,0:2].shape) #map：map(function, element):function on element. 
    print(edges)
    adj = sp.coo_matrix((edges_unordered[:, 2], (edges[:, 0], edges[:, 1])),
                    shape=(idx_features.shape[0], idx_features.shape[0]),
                    dtype=np.float32)
    
    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    #adj = (adj + sp.eye(adj.shape[0])) #diagonal, set to 1
   
    print()
    print(adj)
    return adj

if __name__ == "__main__":
    idx_features = sys.argv[1]
    net_filepath = sys.argv[2]

    print(build_adj_weight(idx_features, net_filepath))