import numpy as np
import pandas as pd
import sys
import math
import tensorflow as tf
from data_preprocessing import get_data
from sklearn import preprocessing
from gcn import gcn


def train(model, train_inputs, train_labels):
    '''
    Trains the model on all of the inputs and labels for one epoch. You should shuffle your inputs 
    and labels - ensure that they are shuffled in the same order using tf.gather or zipping.
    To increase accuracy, you may want to use tf.image.random_flip_left_right on your
    inputs before doing the forward pass. You should batch your inputs.
    
    :param model: the initialized model to use for the forward pass and backward pass
    :param train_inputs: train inputs (all inputs to use for training), 
    shape (num_inputs, width, height, num_channels)
    :param train_labels: train labels (all labels to use for training), 
    shape (num_labels, num_classes)
    :return: Optionally list of losses per batch to use for visualize_loss
    '''

    # Shuffle inputs and labls
    ind = np.arange(0, len(train_inputs))
    tf.random.shuffle(ind)

    train_inputs = tf.gather(train_inputs, ind)
    train_labels = tf.gather(train_labels, ind)

    for i in range(0, int(len(train_inputs)/model.batch_size)):
        # Loop through all the batches
        X_batch = train_inputs[i*model.batch_size:(i+1)*model.batch_size]
        Y_batch = train_labels[i*model.batch_size:(i+1)*model.batch_size]

        X_batch = tf.image.random_flip_left_right(X_batch)

        with tf.GradientTape() as tape:
            
            # Call model.call and model.loss within GradientTape()
            preds = model.call(X_batch)
            loss = model.loss(preds, Y_batch)
            # model.loss_list.append(np.average(loss))

            # if i % 20 == 0:
                # train_acc = model.accuracy(preds, Y_batch)
                # print("Accuracy on training set after {} training steps: {}".format(i, train_acc))

        # Calculate gradients and optimize model 
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))


    return model.loss_list

def test(model, test_inputs, test_labels):
    """
    Tests the model on the test inputs and labels. You should NOT randomly 
    flip images or do any extra preprocessing.
    
    :param test_inputs: test data (all images to be tested), 
    shape (num_inputs, width, height, num_channels)
    :param test_labels: test labels (all corresponding labels),
    shape (num_labels, num_classes)
    :return: test accuracy - this should be the average accuracy across
    all batches
    """

    acc = 0
    
    for i in range(0, int(len(test_inputs)/model.batch_size)):
        # Loop through all the batches
        X_batch = test_inputs[i*model.batch_size:(i+1)*model.batch_size]
        Y_batch = test_labels[i*model.batch_size:(i+1)*model.batch_size]

        # Call model.call, calculate loss and accuracy
        preds = model.call(X_batch, is_testing = True)
        loss = model.loss(preds, Y_batch)
        acc += model.accuracy(preds, Y_batch)

    acc = acc/(int(len(test_inputs)/model.batch_size))
    print("Test accuracy: {}".format(acc))
    return acc


def encode_labels(labels_array):
    label_encoder = preprocessing.LabelEncoder()
    labels = label_encoder.fit_transform(labels_array)
    labels = to_categorical(labels)
    
    return labels, label_encoder.classes_


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



def main(path_exp, path_labels):
    '''
    
    '''
   
    # Process data
    exp_df, adj_matrix, labels_array = get_data(path_exp, path_labels)

    labels_encoded, classes = encode_labels(labels_array)



    # num_epochs = 100
    # for i in range(num_epochs):
    #     print("Epoch:", i)
        # TO DO: Call train

    # TO DO: CTest model!


    return


if __name__ == '__main__':
    path_exp = sys.argv[1]
    path_labels = sys.argv[2]
    main(path_exp, path_labels)
 