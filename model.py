from typing_extensions import final
import numpy as np
import pandas as pd
import sys
import tensorflow as tf
from data_preprocessing import get_data
from sklearn import preprocessing
from gcn import GCN
import matplotlib.pyplot as plt

# Train model
def train(model, train_inputs, train_labels, adj_matrix):
    loss_list = []
    acc_list = []

    for i in range(0, int(len(train_inputs)/model.batch_size)):
        # Loop through all the batches
        X_batch = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        Y_batch = train_labels[i*model.batch_size:(i+1)*model.batch_size]

        with tf.GradientTape() as tape:
            
            # Call model.call and model.loss within GradientTape()
            decoder_pred, final_pred = model.call(adj_matrix, X_batch)

            loss = model.loss(decoder_pred, X_batch, final_pred, Y_batch)
            loss_val = loss.numpy()
    

            if i % 200 == 0:
                print("Loss on training set after {} training steps: {}".format(i, loss))

            # Compute train accuracy
            acc = model.accuracy(final_pred, Y_batch)[0]
            acc_val = acc.numpy()
            print("/ batch accuracy:", acc)
        
        loss_list.append(loss_val)
        acc_list.append(acc_val)

        # Calculate gradients and optimize model 
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

    return sum(loss_list)/len(loss_list), sum(acc_list)/len(acc_list)

# Test model
def test(model, test_inputs, test_labels, adj_matrix):
    acc = 0
    
    for i in range(0, int(len(test_inputs)/model.batch_size)):
        # Loop through all the batches
        X_batch = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        Y_batch = test_labels[i*model.batch_size:(i+1)*model.batch_size]

        # Call model.call, calculate loss and accuracy
        decoder_pred, final_pred = model.call(adj_matrix, X_batch)
        # loss = model.loss(decoder_pred, X_batch, final_pred, Y_batch)
        acc += model.accuracy(final_pred, Y_batch)[0]

    acc = acc/(int(len(test_inputs)/model.batch_size))
    print("Test accuracy: {}".format(acc))
    return acc


def visualize_loss(losses): 
    """
    Uses Matplotlib to visualize the losses of our model.
    :param losses: list of loss data stored from train. Can use the model's loss_list 
    field 

    :return: doesn't return anything, a plot should pop-up 
    """
    x = [i for i in range(len(losses))]
    plt.plot(x, losses)
    plt.title('Loss per batch')
    plt.xlabel('Batch')
    plt.ylabel('Loss')
    plt.show()  


# Main function of model
def main(path_exp, path_labels, preprocess):
   
    if preprocess:
        # Process data
        train_data, val_data, test_data, train_labels, val_labels, test_labels, adj_matrix, num_classes = get_data(path_exp, path_labels)
    else:
        # Load in already processed data
        train_data = np.load("train_data.npy", allow_pickle=True)
        val_data = np.load("val_data.npy", allow_pickle=True)
        test_data = np.load("test_data.npy", allow_pickle=True)
        train_labels = np.load("train_labels.npy", allow_pickle=True)
        val_labels = np.load("val_labels.npy", allow_pickle=True)
        test_labels = np.load("test_labels.npy", allow_pickle=True)

        train_data = tf.convert_to_tensor(train_data, dtype=tf.float32)
        val_data = tf.convert_to_tensor(val_data, dtype=tf.float32)
        test_data = tf.convert_to_tensor(test_data, dtype=tf.float32)
        train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
        val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float32)
        test_labels = tf.convert_to_tensor(test_labels, dtype=tf.float32)

        adj_matrix = pd.read_csv('adj_matrix.csv', index_col=0)
        num_classes = train_labels.shape[1]


    num_genes = adj_matrix.shape[0]
    model = GCN(num_genes, num_classes)
    loss_list = []
    acc_list = []

    print("Training the model...\n")
    num_epochs = 20
    for i in range(num_epochs):
        print("Epoch:", i)
        loss, acc = train(model, train_data, train_labels, adj_matrix)
        loss_list.append(loss)
        acc_list.append(acc)

    # Test model on validation data
    print("Testing the model on validation data...\n")
    accuracy_val = test(model, val_data, val_labels, adj_matrix)
    print("Validation accuracy", accuracy_val)

    # Test model on test data
    print("Testing the model on test data...\n")
    accuracy_test = test(model, test_data, test_labels, adj_matrix)
    print("Test accuracy", accuracy_test)

    # Visualize loss
    print(loss_list)
    print(acc_list)
    print("Visualizing loss...\n")
    visualize_loss(loss_list)
    pass


if __name__ == '__main__':
    path_exp = sys.argv[1]
    path_labels = sys.argv[2]
    main(path_exp, path_labels, preprocess=True)
    # main(path_exp, path_labels, preprocess=False)