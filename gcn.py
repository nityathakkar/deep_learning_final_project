from typing_extensions import final
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn import metrics

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
        self.batch_size = 256
        self.num_genes = num_genes
        self.num_classes = num_classes
        self.pool_size = 8 # Paper specifies this must be a power of 2
        
        # define layers
        self.encoder1 = tf.keras.layers.MaxPool1D(pool_size = self.pool_size)
        self.encoder2 = tf.keras.layers.Flatten()
        self.encoder3 = tf.keras.layers.Dense(32, activation = 'relu')
        self.decoder_layer = tf.keras.layers.Dense(self.num_genes, activation='relu')

        self.gene_exp1 = tf.keras.layers.Dense(256, activation='relu')
        self.gene_exp2 = tf.keras.layers.Dense(32, activation='relu')

        self.final = tf.keras.layers.Dense(self.num_classes, activation='softmax')


    # Define GCN encoder
    def gcn_layer(self, adj_matrix, gene_exp):

        adj_binary = pd.DataFrame(np.where(adj_matrix != 0, 1, 0), index=adj_matrix.index, columns=adj_matrix.columns)
        num_edges = adj_binary.apply(np.sum, axis=0)
        
        adj_array = adj_matrix.to_numpy()
        adj_tensor = tf.convert_to_tensor(adj_array, dtype=tf.float32)
       
        gene_array = gene_exp.numpy()
        gene_exp_T = tf.transpose(gene_array)

        # genes in adj matrix
        N = adj_array.shape[0]
        
        # Create the D matrix with number of edges on the diagonal
        D = tf.zeros((N,N), dtype=tf.float32)
        D = tf.linalg.set_diag(D, num_edges)

        # Create the Laplacian matrix
        L = tf.math.subtract(D, adj_tensor)

        D_negpower = tf.zeros((N,N), dtype=tf.float32)
        
        neg_tensor = tf.fill([N,], -1/2)
        negpower = tf.math.pow(num_edges, neg_tensor)
        negpower = tf.cast(negpower, tf.float32)
        negpower = tf.where(negpower == np.inf, 0, negpower)
        D_negpower = tf.linalg.set_diag(D_negpower, negpower)

        I = tf.eye(N)
       
        matmul1 = tf.linalg.matmul(D_negpower, adj_tensor)
        matmul2 = tf.linalg.matmul(matmul1, D_negpower)

        # Normalize the Laplacian matrix
        L_normalize = tf.math.add(I, matmul2)

        # Eigen decomposition of Laplacian matrix
        eig_val, eig_vec = tf.linalg.eig(L_normalize)

        eig_val = tf.math.real(eig_val)
        eig_val = tf.cast(eig_val, dtype=tf.float32)

        eig_vec = tf.math.real(eig_vec)
        eig_vec = tf.cast(eig_vec, dtype=tf.float32)
        
        K = 5

        # Beta is the parameter we want to learn!!
        beta = tf.Variable(tf.random.truncated_normal([K]))

        max_eigval = tf.reduce_max(eig_val)
        A_tilde = tf.math.subtract(tf.math.divide((2 * eig_val), max_eigval), I)

        A_tilde_npy = A_tilde.numpy()
        beta_npy = beta.numpy()

        # Find chebyshev polynomial coefficients
        h_func = np.polynomial.chebyshev.chebval(A_tilde_npy, beta_npy)

        mat_1 = tf.linalg.matmul(eig_vec, h_func)
        mat_2 = tf.linalg.matmul(mat_1, tf.transpose(eig_vec))

        conv_out = tf.linalg.matmul(mat_2, gene_exp_T)
        conv_out_T = tf.transpose(conv_out)
    
        relu_conv = tf.nn.relu(conv_out_T)
        relu_conv = tf.expand_dims(relu_conv, 2)

        # Max pool, flatten, and pass through dense layer
        out1 = self.encoder1(relu_conv)
        out2 = self.encoder2(out1)
        gcn_out = self.encoder3(out2)

        return gcn_out

    # Call decoder to reconstruct inputs to encoder
    def decoder(self, input):

        return self.decoder_layer(input)

    # Pass gene expression matrix through 2 Dense layers
    def NN_gene_exp(self, gene_exp):
        
        layer1_out = self.gene_exp1(gene_exp)
        return self.gene_exp2(layer1_out)

    # Concatenate output of encoder and NN (gene_exp) to predict final cell types   
    def final_layer(self, encoder_out, NN_gene_exp_out):
        # Concatenate outputs from encoder and NN_gene_exp
        concat = tf.concat([encoder_out, NN_gene_exp_out], axis=1)
        
        # Softmax probabilites   
        return self.final(concat)

    # Forward pass function
    def call(self, adj_matrix, gene_exp):

        # Output shape is batch_size x 32
        gcn_encoder_out = self.gcn_layer(adj_matrix, gene_exp)

        # Num genes x num genes
        decoder_out = self.decoder(gcn_encoder_out)

        # Output shape is batch_size x 32
        NN_gene_exp_out = self.NN_gene_exp(gene_exp)

        # Output shape is batch_size x 5
        final_out = self.final_layer(gcn_encoder_out, NN_gene_exp_out)

        return decoder_out, final_out


    # Loss function
    def loss(self, decoder_pred, encoder_labels, final_pred, final_labels):

        # Compute MSE loss for GCN encoder/decoder
        loss1 = tf.keras.metrics.mean_squared_error(encoder_labels, decoder_pred)
        loss1 = tf.reduce_mean(loss1)

        reg_1 = 1
        reg_2 = 1
        reg_3 = 5e-4

        # Calculate categorical crossentropy between final cell type predictions and labels
        loss2 = tf.keras.metrics.categorical_crossentropy(final_labels, final_pred)

        loss_total = reg_1 * loss1 + reg_2 *  tf.reduce_mean(loss2)

        # Compute trainable parameters loss
        l2_loss = 0
        for param in self.trainable_variables:
            data = param* param
            l2_loss += tf.reduce_sum(data)


        loss3 = (reg_3 * l2_loss)
        loss_total += loss3

        return loss_total

    # Calculate metrics used to record accuracy
    def calculation(self, pred_labels, actual_labels):

        test_acc = metrics.accuracy_score(actual_labels, pred_labels)
        test_f1_macro = metrics.f1_score(actual_labels, pred_labels, average='macro')
        test_f1_micro = metrics.f1_score(actual_labels, pred_labels, average='micro')
        precision = metrics.precision_score(actual_labels, pred_labels, average='micro')
        recall = metrics.recall_score(actual_labels, pred_labels, average='micro')
        # fpr, tpr, _ = metrics.roc_curve(actual_labels, pred_labels)
        # auc = metrics.auc(fpr, tpr)
            
        # print('method','test_acc','f1_test_macro','f1_test_micro','Testprecision','Testrecall')
        # print('GCN', test_acc, test_f1_macro, test_f1_micro, precision,recall)
        return test_acc, test_f1_macro, test_f1_micro, precision, recall
        
    # Calculate  accuracy (defined as # correct predictions/len(labels))
    def accuracy(self, final_pred, final_labels):
        count = 0

        max_prob = tf.argmax(final_pred, axis=1)

        true_ind = tf.argmax(final_labels, axis=1)

        check_equal = tf.equal(max_prob, true_ind)
        check_equal = tf.cast(check_equal, dtype=tf.int32)
        count = tf.reduce_sum(check_equal)

        test_acc, test_f1_macro, test_f1_micro, precision, recall = self.calculation(max_prob, true_ind)
        print("test_acc, ", test_acc, " test_f1_macro, ", test_f1_macro, ", test_f1_micro, ",  test_f1_micro, "precision, ", precision, "recall, ", recall)
        return count/len(final_labels), test_acc, test_f1_macro, test_f1_micro, precision, recall