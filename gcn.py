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
        self.encoder = tf.keras.Sequential([tf.keras.layers.MaxPool1D(pool_size = self.pool_size), tf.keras.layers.Flatten(), tf.keras.layers.Dense(32, activation = 'relu')])

        self.decoder_layer = tf.keras.layers.Dense(self.num_genes, activation='relu')

        # TO DO: change sizes later potentially (these are sizes used in paper)
        self.gene_exp1 = tf.keras.layers.Dense(256, activation='relu')
        self.gene_exp2 = tf.keras.layers.Dense(32, activation='relu')

        # self.final = tf.keras.layers.Dense(self.num_cells, activation='softmax')
        self.final = tf.keras.layers.Dense(self.num_classes, activation=tf.nn.log_softmax)



    def gcn_layer(self, adj_matrix, gene_exp):
        adj_binary = pd.DataFrame(np.where(adj_matrix != 0, 1, 0), index=adj_matrix.index, columns=adj_matrix.columns)
        num_edges = adj_binary.apply(np.sum, axis=1)
        
        adj_array = adj_matrix.to_numpy()
        adj_tensor = tf.convert_to_tensor(adj_array, dtype=tf.float32)
        gene_array = gene_exp.numpy()
        gene_exp_T = tf.transpose(gene_array)

        N = adj_array.shape[0]

        # for each row, find number of non-zero values
        # convert adj matrix where if 0 --> 0, if not zero --> 1
        # calcualte row sum
        D = tf.zeros((N,N), dtype=tf.float32)
        tf.linalg.set_diag(D, num_edges)

        I = np.eye(N)

        L = tf.math.subtract(D,adj_tensor)

        L = I + tf.linalg.matmul(tf.math.pow(D, -1/2), tf.linalg.matmul(adj_tensor, tf.math.pow(D, 1/2)))


        # eig_vec, eig_val, eig_vec_T = np.linalg.svd(L)
        print(L == tf.transpose(L))
        eig_val, eig_vec = tf.linalg.eig(L)

        K = 5
        beta = tf.Variable(tf.random.truncated_normal([K]))

        max_eigval = max(eig_val)
        A_tilde = tf.math.subtract(tf.math.divide((2 * eig_val), max_eigval), I)

        # h_func = np.polynomial.chebyshev.chebval(A_tilde, beta)
        h_func = tf.math.polyval(beta, A_tilde)

        conv_out = tf.linalg.matmul(tf.linalg.matmul(tf.linalg.matmul(eig_vec, h_func), eig_vec.T), gene_exp_T)

        softmax_conv = tf.nn.softmax(conv_out)

        gcn_out = self.encoder(softmax_conv)

        return gcn_out

    def decoder(self, input):
        # USE MSE AS LOSS FUNCTION
        return self.decoder_layer(input)

    def NN_gene_exp(self, gene_exp):
        
        layer1_out = self.gene1(gene_exp)
        return self.gene2(layer1_out)

        
    def final_layer(self, encoder_out, NN_gene_exp_out):
        # concatenate outputs from encoder and NN_gene_exp
        concat = tf.concat([encoder_out, NN_gene_exp_out], axis=0)
        
        # softmax probabilites   
        return self.final(concat)

    
    def call(self, adj_matrix, gene_exp):
        print("here 1")
        gcn_encoder_out = self.gcn_layer(adj_matrix, gene_exp)

        print("here 2")
        decoder_out = self.decoder(gcn_encoder_out)

        print("here 3")
        NN_gene_exp_out = self.NN_gene_exp(gene_exp)

        print("here 4")
        final_out = self.final_layer(gcn_encoder_out, NN_gene_exp_out)

        print("about to return")
        return decoder_out, final_out


    def loss(self, decoder_pred, encoder_labels, final_pred, final_labels):

        loss1 = tf.keras.metrics.mean_squared_error(encoder_labels, decoder_pred)
        loss1 = tf.reduce_mean(loss1)

        reg_1 = 1
        reg_2 = 1
        reg_3 = 5e-4


        loss2 = -1 * tf.reduce_sum(tf.math.log(tf.dot(final_pred, final_labels)))
        loss_total = 1 * loss1 + 1 * loss2 

        l2_loss = 0
        for param in self.parameters():
            data = param* param
            l2_loss += tf.reduce_sum(data)


        loss_total += (reg_3 * l2_loss)

        return loss_total