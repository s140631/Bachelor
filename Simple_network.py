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
import os
from tensorflow.core.protobuf import saver_pb2

## Exstract data from function

path = "No_fault_data"
Model_no = 40
files = [f for f in os.listdir(path)]
for f in files:
    tf.reset_default_graph()
    sess = tf.Session()
    COLUMNS = ["Time", "StationID", "MainBearingGTemp", "MainBearingHTemp", "GearOilTemp", "AmbientTemp",
               "NacelleTemp", "ActivePower", "GenRPM"]
    data = pd.read_csv(path + "/" + f, names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
    data.Time = pd.to_datetime(data.Time)

    X_train, Y_train, X_test, Y_test, X_val, Y_val = Data_processing.define_data(data, 24)


    activation_functions = tf.nn.sigmoid #best choice by validation, using LR = 0.001, training_epochs = 1000, batch_size = 1000
    LAMBDA = 0.0001
    # Parameters
    momentum = 0.9
    learning_rate = 0.001
    training_epochs = 1000
    display_step = 1000
    batch_size = 1000

    LAMBDA_hid = LAMBDA
    LAMBDA_out = LAMBDA_hid

    # Network parameters
    input_size = X_train.shape[1]
    hidden_size = 1
    hidden_units = 256 ## Found to be the best
    output_size = 1

    # define placeholders
    X = tf.placeholder(tf.float32, [None, input_size])
    Y = tf.placeholder(tf.float32, [None, output_size])

    W_hidden = tf.Variable(tf.random_normal([input_size, hidden_units]), name = "Hidden_weights")
    W_output = tf.Variable(tf.random_normal([hidden_units, output_size]), name = "Output_weights")

    saver = tf.train.Saver(write_version=saver_pb2.SaverDef.V1)

    def multilayer_perceptron(x, W_hidden, W_output):
        # Hidden layer with RELU activation
        hidden_layer = tf.matmul(x, W_hidden)
        hidden_layer = tf.sigmoid(hidden_layer)
        # Output layer with linear activation
        out_layer = tf.matmul(hidden_layer, W_output)
        return out_layer


    pred = multilayer_perceptron(X, W_hidden, W_output)

    # Define loss and optimizer
    cost = tf.losses.mean_squared_error(labels = Y, predictions = pred) + tf.reduce_sum(LAMBDA_hid*tf.nn.l2_loss(W_hidden)) + tf.reduce_sum(LAMBDA_out*tf.nn.l2_loss(W_output))
    #cost = tf.losses.mean_squared_error(labels = Y, predictions = pred)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum= momentum).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    # Launch the graph
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
        save_path = saver.save(sess, os.path.join(os.getcwd(), 'SimpleNetwork'), global_step=Model_no)
        print("Model saved in file: %s" % save_path)
        print("Optimization Finished!")

        PREDICTIONS_train = multilayer_perceptron(tf.cast(X_train, tf.float32), W_hidden, W_output)
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(Y_train - PREDICTIONS_train)))
        print("Training error" + str(sess.run(RMSE)))

        PREDICTIONS_test = multilayer_perceptron(tf.cast(X_test, tf.float32), W_hidden, W_output)
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(Y_test - PREDICTIONS_test)))
        print("Test error" + str(sess.run(RMSE)))
        Model_no = Model_no + 1

