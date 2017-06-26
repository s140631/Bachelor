for name in dir():
    if not name.startswith('_'):
        del globals()[name]


import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from datetime import datetime, timedelta
import pandas as pd
import random
from itertools import islice
from Data_management import Data_processing
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import os
from tensorflow.core.protobuf import saver_pb2
## Exstract data from function

path = "No_fault_data/"
Model_no = 40
files = [f for f in os.listdir(path)]
for f in files:
    tf.reset_default_graph()
    sess = tf.Session()
    COLUMNS = ["Time", "StationID", "MainBearingGTemp", "MainBearingHTemp", "GearOilTemp", "AmbientTemp",
               "NacelleTemp", "ActivePower", "GenRPM"]
    data = pd.read_csv(path + f, names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
    data.Time = pd.to_datetime(data.Time)

    X_train, Y_train, X_test, Y_test, X_val, Y_val = Data_processing.define_data(data, 24)

    momentum = 0.9
    learning_rate = 0.001
    training_epochs = 10000 #set this 10000
    display_step = 1000
    batch_size = 1000

    LAMBDA_hid = 0.001
    LAMBDA_out_one = LAMBDA_hid
    LAMBDA_out_two = LAMBDA_hid

    # Network parameters
    input_size = X_train.shape[1]
    hidden_size = 1
    hidden_units = 256
    output_size = 1

    # define placeholders
    X = tf.placeholder(tf.float32, [None, input_size])
    Y = tf.placeholder(tf.float32, [None, output_size])

    W_hid = tf.Variable(tf.random_normal([input_size, hidden_units],1),name="Hidden_weights")
    W_out_one = tf.Variable(tf.random_normal([hidden_units, 1]), name = "W_out_one")
    W_out_two = tf.Variable(tf.random_normal([hidden_units, 1]), name = "W_out_two")

    # Adding Saver object
    saver = tf.train.Saver(write_version = saver_pb2.SaverDef.V1)

    def multilayer_perceptron(x, w_hid, w_out_one, w_out_two):
        # Hidden layer with RELU activation
        hidden_layer = tf.matmul(x, w_hid)
        hidden_layer = tf.nn.sigmoid(hidden_layer)
        # Output layer with linear activation
        out_layer_one = tf.matmul(hidden_layer, w_out_one)
        out_layer_two = tf.matmul(hidden_layer, w_out_two)
        return out_layer_one, out_layer_two

    a_mu, a_sig = multilayer_perceptron(X, W_hid, W_out_one, W_out_two)


    # Define loss and optimizer
    cost = 1/2 * tf.reduce_sum(tf.losses.mean_squared_error(labels = Y, predictions = a_mu)/tf.exp(a_sig)) + 1/2* tf.reduce_sum(tf.log(tf.exp(a_sig))) + 1/2 * tf.reduce_sum(LAMBDA_hid*tf.nn.l2_loss(W_hid) + LAMBDA_out_one*tf.nn.l2_loss(W_out_one) + LAMBDA_out_two*tf.nn.l2_loss(W_out_two))
    #cost = tf.losses.mean_squared_error(labels = Y, predictions = pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum= momentum).minimize(cost)


    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
    #sess = tf.Session()

    with tf.Session() as sess:
        sess.run(init)
        for epoch in range(training_epochs):
            # Divide into batches
            total_batches = int(len(X_train) / batch_size)
            for i in range(total_batches + 1):
                if i != total_batches:
                    batch_x = X_train[i * batch_size:(i + 1) * batch_size, :]
                    batch_y = Y_train[i * batch_size:(i + 1) * batch_size, :]
                else:
                    batch_x = X_train[i * batch_size:, :]
                    batch_y = Y_train[i * batch_size:, :]
                sess.run(optimizer,feed_dict={X: batch_x, Y: batch_y})
            if epoch % display_step == 0:
                print(sess.run(cost, feed_dict={X: X_train, Y: Y_train}))

        save_path = saver.save(sess, os.path.join(os.getcwd(),'NetworkWithVariance'), global_step=Model_no)
        print("Model saved in file: %s" % save_path)
        print("Optimization Finished!")
        PREDICTIONS_train, _ = multilayer_perceptron(tf.cast(X_train, tf.float32), W_hid, W_out_one, W_out_two)
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(Y_train - PREDICTIONS_train)))
        print("Training error" + str(sess.run(RMSE)))

        PREDICTIONS_test, _ = multilayer_perceptron(tf.cast(X_test, tf.float32), W_hid, W_out_one, W_out_two)
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(Y_test - PREDICTIONS_test)))
        print("Test error" + str(sess.run(RMSE)))
        Model_no = Model_no + 1




    #RMSE_validation[loop] = sess.run(tf.sqrt(tf.reduce_mean(tf.square(Y_val - multilayer_perceptron(tf.cast(X_val, tf.float32), weights)))))
    #loop = loop+1
    #plt.bar(range(len(RMSE_validation)), RMSE_validation, 0.35, color="red")
    #plt.xticks(range(len(RMSE_validation)), act_func_labels, size='small')
    #plt.ylabel("Root mean square error")
    #plt.title("Choice of activation function")
    #plt.show()
