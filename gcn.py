from typing_extensions import final
import numpy as np
import pandas as pd
import tensorflow as tf

# encoder part of the GCN model consists of a GCN layer followed by a maxpooling layer, 
# a flatten layer and a fully connected (FC) layer. While the decoder part consists of a 
# FC layer to reconstruct the gene expression values.

class GCN(tf.keras.Model):
    def __init__(self, num_genes, num_classes):
    # define all the layers, hyperparameters
        super(GCN, self).__init__()

        # define hyperparameters
        self.learning_rate = 1e-3
        self.optimizer = tf.keras.optimizers.Adam(self.learning_rate)
        self.batch_size = 128
        self.num_genes = num_genes
        self.num_classes = num_classes
        self.pool_size = 8 # Paper specifies this must be a power of 2
        
        # define layers
        # self.encoder = tf.keras.Sequential([tf.keras.layers.MaxPool1D(pool_size = self.pool_size), tf.keras.layers.Flatten(), tf.keras.layers.Dense(32, activation = 'relu')])

        self.encoder1 = tf.keras.layers.MaxPool1D(pool_size = self.pool_size)
        self.encoder2 = tf.keras.layers.Flatten()
        self.encoder3 = tf.keras.layers.Dense(32, activation = 'relu')
        self.decoder_layer = tf.keras.layers.Dense(self.num_genes, activation='relu')

        # TO DO: change sizes later potentially (these are sizes used in paper)
        self.gene_exp1 = tf.keras.layers.Dense(256, activation='relu')
        self.gene_exp2 = tf.keras.layers.Dense(32, activation='relu')

        # self.final = tf.keras.layers.Dense(self.num_cells, activation='softmax')
        self.final = tf.keras.layers.Dense(self.num_classes, activation=tf.nn.softmax)



    def gcn_layer(self, adj_matrix, gene_exp):

        # adj_matrix_genes = adj_matrix.columns
        # print(len(adj_matrix_genes))
        # gene_exp_genes = gene_exp.columns
        # print(len(gene_exp.shape))

        # gene_intersect = list(set(adj_matrix_genes) & set(gene_exp_genes))

        # print(len(gene_intersect))



        adj_binary = pd.DataFrame(np.where(adj_matrix != 0, 1, 0), index=adj_matrix.index, columns=adj_matrix.columns)
        # print(adj_binary)
        num_edges = adj_binary.apply(np.sum, axis=0)

        adj_binary_npy = adj_binary.values
        adj_binary_tensor = tf.convert_to_tensor(adj_binary_npy, dtype=tf.float32)
        adj_bin_df = pd.DataFrame(adj_binary_npy)
        adj_bin_df.to_csv('adj_Binary.csv')
        
        # print(num_edges)
        adj_array = adj_matrix.to_numpy()
        # print(np.all(adj_array == adj_array.T))  
        # print(np.any(adj_array != adj_array.T))
        adj_tensor = tf.convert_to_tensor(adj_array, dtype=tf.float32)
        
       
       
        gene_array = gene_exp.numpy()
        # print(gene_array.shape)
        gene_exp_T = tf.transpose(gene_array)

        # num genes x num cells
        # print(gene_exp_T.shape)

        # genes in adj matrix
        N = adj_array.shape[0]
        # print(N)

        # for each row, find number of non-zero values
        # convert adj matrix where if 0 --> 0, if not zero --> 1
        # calcualte row sum
        
        D = tf.zeros((N,N), dtype=tf.float32)
        D = tf.linalg.set_diag(D, num_edges)
        D_alone_npy = D.numpy()
        D_alone_df = pd.DataFrame(D_alone_npy)
        D_alone_df.to_csv("D.csv")

        L = tf.math.subtract(D, adj_tensor)

        D_negpower = tf.zeros((N,N), dtype=tf.float32)
        # D_pospower = tf.zeros((N,N), dtype=tf.float32)
        
        # print(num_edges.shape)
        neg_tensor = tf.fill([N,], -1/2)
        negpower = tf.math.pow(num_edges, neg_tensor)
        negpower = tf.cast(negpower, tf.float32)
        negpower = tf.where(negpower == np.inf, 0, negpower)
        D_negpower = tf.linalg.set_diag(D_negpower, negpower)
        D_negpower_npy = D_negpower.numpy()
        D_df = pd.DataFrame(D_negpower_npy)
        D_df.to_csv("D_matrix.csv")


        # print(tf.reduce_all(tf.math.equal(tf.linalg.matmul(D_negpower, L), tf.transpose(tf.linalg.matmul(D_negpower, L)))))

        I = tf.eye(N)
        matmul1 = tf.linalg.matmul(D_negpower, adj_tensor)
        # matmul1_npy = matmul1.numpy()
        # matmul1_df = pd.DataFrame(matmul1_npy)
        # matmul1_df.to_csv("matmul1_matrix.csv")


        matmul2 = tf.linalg.matmul(matmul1, D_negpower)
        # matmul2_npy = matmul2.numpy()
        # matmul2_df = pd.DataFrame(matmul2_npy)
        # matmul2_df.to_csv("matmul2_matrix.csv")

        L_normalize = tf.math.add(I, matmul2)
        L_normalize_npy = L_normalize.numpy()
        L_df = pd.DataFrame(L_normalize_npy)
        L_df.to_csv("normalized_laplacian.csv")
        # L_normalize = tf.matmul(tf.matmul(D_negpower, L), D_negpower)
        # print(tf.reduce_all(tf.math.equal(L_normalize, tf.transpose(L_normalize))))
        # print(tf.reduce_any(tf.math.not_equal(L_normalize, tf.transpose(L_normalize))))
        # print(tf.reduce_sum(tf.where(tf.math.equal(L_normalize, tf.transpose(L_normalize)) == False, 1, 0)))



        # L_normalize = tf.math.subtract
        # I = np.eye(N)
       


        # pos_tensor = tf.fill([N,], 1/2)
        # pospower = tf.math.pow(num_edges, pos_tensor)
        # pospower = tf.cast(pospower, tf.float32)
        # D_pospower = tf.linalg.set_diag(D_pospower, pospower)
        # print(num_edges)
        # print(D_negpower)
        # print(D_pospower)

        # D_power = 
        # # print(D)

        

        # print(tf.math.pow(D, -1/2))
        # print(tf.math.pow(D, 1/2))
        # print(tf.linalg.matmul(adj_tensor, tf.math.pow(D, 1/2)))
        # print(adj_tensor)
        # print(tf.linalg.matmul(adj_tensor, D_pospower))
        # print(tf.linalg.matmul(D_negpower, tf.linalg.matmul(adj_tensor, D_pospower)))

        # print(adj_tensor)
        # print(D_pospower)
        # print(D_negpower)
        # print(tf.linalg.matmul(adj_tensor, D_pospower))

        # L = I + tf.linalg.matmul(tf.linalg.matmul(adj_tensor, D_negpower), D_negpower)
        # print(tf.reduce_sum(tf.where(tf.math.equal(L, tf.transpose(L)) == False, 1, 0)))
        # print(tf.reduce_all(tf.math.equal(L, tf.transpose(L))))
        # print(L.T)

        # print(L)
        # print(tf.transpose(L))
        # print(tf.math.equal(L, tf.transpose(L)))
        # # print(.shape)
        


        # eig_vec, eig_val, eig_vec_T = np.linalg.svd(L)
        # print(L == tf.transpose(L))
        L_test = (L_normalize + tf.transpose(L_normalize))/2
        eig_val, eig_vec = tf.linalg.eig(L_test)
        # eig_val, eig_vec = tf.linalg.eig(L_normalize)
        # print(eig_val)
        eig_val = tf.math.real(eig_val)
        eig_val = tf.cast(eig_val, dtype=tf.float32)

        eig_vec = tf.math.real(eig_vec)
        eig_vec = tf.cast(eig_vec, dtype=tf.float32)
        
        K = 5
        beta = tf.Variable(tf.random.truncated_normal([K]))
        # print(beta)
        # print(beta.shape)

        # max_eigval = max(eig_val)

        max_eigval = tf.reduce_max(eig_val)
        A_tilde = tf.math.subtract(tf.math.divide((2 * eig_val), max_eigval), I)

        A_tilde_npy = A_tilde.numpy()
        beta_npy = beta.numpy()

        h_func = np.polynomial.chebyshev.chebval(A_tilde_npy, beta_npy)
        # print(h_func)
        # print(h_func.shape)
        # h_func = tf.math.polyval(beta, A_tilde)
        # print(eig_vec.shape)
        # print(h_func.shape)
        # print(tf.transpose(eig_vec).shape)
        # print(gene_exp_T.shape)

        mat_1 = tf.linalg.matmul(eig_vec, h_func)
        mat_2 = tf.linalg.matmul(mat_1, tf.transpose(eig_vec))
        # print(mat_2)
        # print(gene_exp_T)

        # gene_exp_genes = gene_exp.index
        # print(gene_exp_genes)

        # adj_matrix_genes = adj_matrix.columns
        # print(adj_matrix_genes)

        # gene_intersect = list(set(lst1) & set(lst2))


        conv_out = tf.linalg.matmul(mat_2, gene_exp_T)
        conv_out_T = tf.transpose(conv_out)
    
        relu_conv = tf.nn.relu(conv_out_T)
        relu_conv = tf.expand_dims(relu_conv, 2)
        # print(relu_conv)
        # num genes x batch size (aka num cells)
        # print(relu_conv.shape)
        out1 = self.encoder1(relu_conv)
        out2 = self.encoder2(out1)
        gcn_out = self.encoder3(out2)
        # print(gcn_out)
        # print(gcn_out.shape)
        # print(L.T.shape)

        return gcn_out

    def decoder(self, input):
        # USE MSE AS LOSS FUNCTION
        return self.decoder_layer(input)

    def NN_gene_exp(self, gene_exp):
        
        layer1_out = self.gene_exp1(gene_exp)
        return self.gene_exp2(layer1_out)

        
    def final_layer(self, encoder_out, NN_gene_exp_out):
        # concatenate outputs from encoder and NN_gene_exp
        concat = tf.concat([encoder_out, NN_gene_exp_out], axis=1)
        
        # softmax probabilites   
        return self.final(concat)

    
    def call(self, adj_matrix, gene_exp):
        print("here 1")
        gcn_encoder_out = self.gcn_layer(adj_matrix, gene_exp)
        print(gcn_encoder_out.shape) # #genes x 32

        print("here 2")
        decoder_out = self.decoder(gcn_encoder_out)
        print(decoder_out.shape) # #genes x #genes

        print("here 3")
        NN_gene_exp_out = self.NN_gene_exp(gene_exp)
        print(NN_gene_exp_out.shape) # batch size x 32

        print("here 4")
        final_out = self.final_layer(gcn_encoder_out, NN_gene_exp_out)
        print(final_out.shape) #batch size + num_genes x 5

        print("about to return")
        return decoder_out, final_out


    def loss(self, decoder_pred, encoder_labels, final_pred, final_labels):

        loss1 = tf.keras.metrics.mean_squared_error(encoder_labels, decoder_pred)
        loss1 = tf.reduce_mean(loss1)
        print("loss 1", loss1)

        reg_1 = 1
        reg_2 = 1
        reg_3 = 5e-4

        # pred_ind = np.arange(len(final_pred))
        # pred_input = np.array([pred_ind, final_pred]).T
        indices = (np.arange(len(final_labels)), final_labels)
        indices = tf.cast(indices, tf.int32)

        indices = tf.stack(indices, axis=1)

        probs = tf.gather_nd(final_pred, indices)
        print(probs)

        # final_labels = tf.cast(final_labels, tf.int32)

        # prob_labels = tf.gather_nd(final_labels, probs)

        # print(final_labels)
        # print(final_pred)
        #prob_labels = final_pred[:, final_labels]
        loss2 = -1 * tf.reduce_sum(tf.math.log(probs))
        print("loss 2", loss2)
        loss_total = reg_1 * loss1 + reg_2 * loss2 

        l2_loss = 0
        for param in self.trainable_variables:
            data = param* param
            l2_loss += tf.reduce_sum(data)


        loss3 = (reg_3 * l2_loss)
        print("loss 3", loss3)
        loss_total += loss3

        return loss_total