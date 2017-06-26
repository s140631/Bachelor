## THIS SCRIPT CONTAINS THE CODE USED FOR TRAINING THE LINEAR MODEL

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
## Exstract data from function


LM_RMSE_train = []
LM_RMSE_test = []
Model_no = 1
path = "Data"
files = [f for f in os.listdir(path)]
for f in files:

    COLUMNS = ["Time", "StationID", "MainBearingGTemp", "MainBearingHTemp", "GearOilTemp", "AmbientTemp",
               "NacelleTemp", "ActivePower", "GenRPM"]
    data = pd.read_csv("Data/" + f, names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
    data.Time = pd.to_datetime(data.Time)

    X_train, Y_train, X_test, Y_test, X_val, Y_val = Data_processing.define_data(data, 24)

    #X_train, Y_train, X_test, Y_test, X_val, Y_val = Data_processing.load_data("Data", 24)

    DIMENSIONS = X_train.shape[1]
    DS_SIZE = X_train.shape[0]
    LAMBDA = 1  # L2 regularization factor
    graph = tf.Graph()
    with graph.as_default():
      # declare graph inputs
      X = tf.placeholder(tf.float32, shape=(DS_SIZE, DIMENSIONS ))
      Y = tf.placeholder(tf.float32, shape=(DS_SIZE, 1))
      theta = tf.Variable([[0.0] for _ in range(DIMENSIONS + 1)])  # implicit bias!
      # optimum
      optimum = tf.matrix_solve_ls(X, Y, LAMBDA, fast=True)

    # run the computation: no loop needed!
    with tf.Session(graph=graph) as s:
      tf.initialize_all_variables().run()
      print("initialized")
      opt = s.run(optimum, feed_dict={X: X_train, Y: Y_train})
      W = opt
      #print("Solution for parameters:\n", W)

    np.save("Weights" +str(Model_no), W)
    Model_no = Model_no + 1

    ## Training error

    PREDICTIONS_train = np.dot(X_train,W)
    print(PREDICTIONS_train[0])
    print(Y_train[0])

    RMSE = np.sqrt(np.mean(np.square(PREDICTIONS_train - Y_train)))
    print("Training error " + str(RMSE))

    ## Test error
    PREDICTIONS_test = np.dot(X_test,W)
    print(PREDICTIONS_test[0])
    print(Y_train[0])

    RMSE = np.sqrt(np.mean(np.square(PREDICTIONS_test - Y_test)))
    print("Test error " + str(RMSE))

    LM_RMSE_train = np.append(LM_RMSE_train, PREDICTIONS_train - Y_train)
    LM_RMSE_test = np.append(LM_RMSE_test, PREDICTIONS_test - Y_test,0)


RMSE_train = np.sqrt(np.mean(np.square(LM_RMSE_train)))
RMSE_test = np.sqrt(np.mean(np.square(LM_RMSE_test)))



