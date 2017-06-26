import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import *
import tensorflow as tf
from datetime import datetime, timedelta
import pandas as pd
import random
from itertools import islice
from Data_management import Data_processing
import os
from sklearn import metrics
import scikitplot.plotters as skplt
## Exstract data from function

def no_failure_data(data, n, failure_date):
    failure_date = pd.to_datetime(failure_date)
    length_of_window = n
    data = data[~pd.isnull(data.StationID)]
    data = data[~pd.isnull(data.MainBearingGTemp)]
    data = data[~pd.isnull(data.MainBearingHTemp)]
    data = data[~pd.isnull(data.GearOilTemp)]
    data = data[~pd.isnull(data.AmbientTemp)]
    data = data[~pd.isnull(data.NacelleTemp)]
    data = data[~pd.isnull(data.ActivePower)]
    data = data[~pd.isnull(data.GenRPM)]

    ### Filters!

    data = data.drop(data[data.ActivePower < 1000].index)
    data = data.reset_index()

    ##### The first thing we want to do, is to normalize the data
    data.GearOilTemp = (data.GearOilTemp - np.mean(data.GearOilTemp)) / np.std(data.GearOilTemp)
    data.AmbientTemp = (data.AmbientTemp - np.mean(data.AmbientTemp)) / np.std(data.AmbientTemp)
    data.NacelleTemp = (data.NacelleTemp - np.mean(data.NacelleTemp)) / np.std(data.NacelleTemp)
    m = 0.5 * (data.ActivePower.max(skipna=True) + data.ActivePower.min(skipna=True))
    s = 0.5 * (data.ActivePower.max(skipna=True) - data.ActivePower.min(skipna=True))
    data.ActivePower = (data.ActivePower - m) / s
    m = 0.5 * (data.GenRPM.max(skipna=True) + data.GenRPM.min(skipna=True))
    s = 0.5 * (data.GenRPM.max(skipna=True) - data.GenRPM.min(skipna=True))
    data.GenRPM = (data.GenRPM - m) / s

    #### Training, test and validation set. Training data is extracted from assumed no-fault states in the first 14 months of operation. However, the
    ### first month is deleted, to make up for commisionning and testing

    data['labels'] = pd.Series(0, index=data.index)

    training_data = data
    # start = training_data.Time[0] + timedelta(days=60)
    start = training_data.Time[0] + timedelta(days=60)
    training_data = training_data.drop(training_data[training_data.Time < start].index)  # Delete the first months
    training_data = training_data.reset_index()
    del training_data['index']

    ### Filters!

    X_test = training_data[["GearOilTemp", "AmbientTemp", "NacelleTemp", "ActivePower", "GenRPM"]]
    Y_test = training_data[["MainBearingHTemp", "labels"]]
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    ##### Establish sliding window!
    n = length_of_window
    X_test = Data_processing.Sliding_window(X_test, n)
    Y_test = np.delete(Y_test, [range(n)], 0)

    # Adding bias term to X
    X_BIAS = np.ones((X_test.shape[0], X_test.shape[1] + 1))
    X_BIAS[:, :-1] = X_test
    X_test = X_BIAS

    # Test and validation set
    return X_test, Y_test

def failure_data(data, n, failure_date, deltaT, fail):
    failure_date = pd.to_datetime(failure_date)
    length_of_window = n
    data = data[~pd.isnull(data.StationID)]
    data = data[~pd.isnull(data.MainBearingGTemp)]
    data = data[~pd.isnull(data.MainBearingHTemp)]
    data = data[~pd.isnull(data.GearOilTemp)]
    data = data[~pd.isnull(data.AmbientTemp)]
    data = data[~pd.isnull(data.NacelleTemp)]
    data = data[~pd.isnull(data.ActivePower)]
    data = data[~pd.isnull(data.GenRPM)]

    ### Filters!

    data = data.drop(data[data.ActivePower < 1200].index)
    data = data.reset_index()

    ##### The first thing we want to do, is to normalize the data
    data.GearOilTemp = (data.GearOilTemp - np.mean(data.GearOilTemp)) / np.std(data.GearOilTemp)
    data.AmbientTemp = (data.AmbientTemp - np.mean(data.AmbientTemp)) / np.std(data.AmbientTemp)
    data.NacelleTemp = (data.NacelleTemp - np.mean(data.NacelleTemp)) / np.std(data.NacelleTemp)
    m = 0.5 * (data.ActivePower.max(skipna=True) + data.ActivePower.min(skipna=True))
    s = 0.5 * (data.ActivePower.max(skipna=True) - data.ActivePower.min(skipna=True))
    data.ActivePower = (data.ActivePower - m) / s
    m = 0.5 * (data.GenRPM.max(skipna=True) + data.GenRPM.min(skipna=True))
    s = 0.5 * (data.GenRPM.max(skipna=True) - data.GenRPM.min(skipna=True))
    data.GenRPM = (data.GenRPM - m) / s

    #### Training, test and validation set. Training data is extracted from assumed no-fault states in the first 14 months of operation. However, the
    ### first month is deleted, to make up for commisionning and testing

    data['labels'] = pd.Series(0, index=data.index)
    if fail == True:
        data['labels'] = 1

    # data['labels'].loc[data.Time > failure_date] = 1

    training_data = data
    #start = training_data.Time[0] + timedelta(days=60)
    start = failure_date - timedelta(days = deltaT)
    training_data = training_data.drop(training_data[training_data.Time < start].index)  # Delete the first months
    training_data = training_data.drop(training_data[training_data.Time > failure_date].index)
    training_data = training_data.reset_index()
    del training_data['index']

    ### Filters!

    X_test = training_data[["GearOilTemp", "AmbientTemp", "NacelleTemp", "ActivePower", "GenRPM"]]
    Y_test = training_data[["MainBearingHTemp", "labels"]]
    X_test = np.asarray(X_test)
    Y_test = np.asarray(Y_test)

    ##### Establish sliding window!
    n = length_of_window
    X_test = Data_processing.Sliding_window(X_test, n)
    Y_test = np.delete(Y_test, [range(n)], 0)

    # Adding bias term to X
    X_BIAS = np.ones((X_test.shape[0], X_test.shape[1] + 1))
    X_BIAS[:, :-1] = X_test
    X_test = X_BIAS

    # Test and validation set
    return X_test, Y_test

def LinearModel(X_input, Y_input):
    tf.reset_default_graph()
    DIMENSIONS = X_input.shape[1]
    DS_SIZE = X_input.shape[0]
    LAMBDA = 1  # L2 regularization factor
    graph = tf.Graph()
    with graph.as_default():
        # declare graph inputs
        X = tf.placeholder(tf.float32, shape=(DS_SIZE, DIMENSIONS))
        Y = tf.placeholder(tf.float32, shape=(DS_SIZE, 1))
        theta = tf.Variable([[0.0] for _ in range(DIMENSIONS + 1)])  # implicit bias!
        # optimum
        optimum = tf.matrix_solve_ls(X, Y, LAMBDA, fast=True)

    # run the computation: no loop needed!
    with tf.Session(graph=graph) as s:
        tf.initialize_all_variables().run()
        print("initialized")
        opt = s.run(optimum, feed_dict={X: X_input, Y: np.expand_dims(Y_input[:,0],1)})
        W = opt
        # print("Solution for parameters:\n", W)

    return W

n = 24 # sliding window length


COLUMNS = ["Time", "StationID", "MainBearingGTemp", "MainBearingHTemp", "GearOilTemp", "AmbientTemp",
               "NacelleTemp", "ActivePower", "GenRPM"]
data1 = pd.read_csv("No_fault_data/A03.csv", names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
data1.Time = pd.to_datetime(data1.Time)
data2 = pd.read_csv("No_fault_data/A04.csv", names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
data2.Time = pd.to_datetime(data2.Time)
data3 = pd.read_csv("No_fault_data/A05.csv", names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
data3.Time = pd.to_datetime(data3.Time)
data4 = pd.read_csv("No_fault_data/B03.csv", names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
data4.Time = pd.to_datetime(data4.Time)


X_A03, Y_A03 , _, _, _, _ = Data_processing.define_data(data1, 24)
X_A04, Y_A04 , _, _, _, _ = Data_processing.define_data(data2, 24)
X_A05, Y_A05 , _, _, _, _ = Data_processing.define_data(data3, 24)
X_B03, Y_B03 , _, _, _, _ = Data_processing.define_data(data4, 24)

W_A03 = LinearModel(X_A03, Y_A03)
W_A04 = LinearModel(X_A04, Y_A04)
W_A05 = LinearModel(X_A05, Y_A05)
W_B03 = LinearModel(X_B03, Y_B03)

X_A03_test, Y_A03_test = no_failure_data(data1, n, 20)
X_A04_test, Y_A04_test = no_failure_data(data2, n, 20)
X_A05_test, Y_A05_test = no_failure_data(data3, n, 20)
X_B03_test, Y_B03_test = no_failure_data(data4, n, 20)

sess = tf.Session()
pred_A03 = np.dot(X_A03_test, W_A03)
Res_A03 = np.expand_dims(Y_A03_test[:,0],1) - pred_A03
pred_A04 = np.dot(X_A04_test, W_A04)
Res_A04 = np.expand_dims(Y_A04_test[:,0],1) - pred_A04
pred_A05 = np.dot(X_A05_test, W_A05)
Res_A05 = np.expand_dims(Y_A05_test[:,0],1) - pred_A05
pred_B03 = np.dot(X_B03_test, W_B03)
Res_B03 = np.expand_dims(Y_B03_test[:,0],1) - pred_B03

Res_A03 = Data_processing.Sliding_window(Res_A03, 5)
Res_A03 = np.expand_dims(np.mean(Res_A03, 1), 1)
Res_A04 = Data_processing.Sliding_window(Res_A04, 5)
Res_A04 = np.expand_dims(np.mean(Res_A04, 1), 1)
Res_A05 = Data_processing.Sliding_window(Res_A05, 5)
Res_A05 = np.expand_dims(np.mean(Res_A05, 1), 1)
Res_B03 = Data_processing.Sliding_window(Res_B03, 5)
Res_B03 = np.expand_dims(np.mean(Res_B03, 1), 1)

Results_G01 = np.append(Res_A03, np.expand_dims(Y_A03_test[5:, 1], 1), 1)
Results_G02 = np.append(Res_A04, np.expand_dims(Y_A04_test[5:, 1], 1), 1)
Results_G03 = np.append(Res_A05, np.expand_dims(Y_A05_test[5:, 1], 1), 1)
Results_G04 = np.append(Res_B03, np.expand_dims(Y_B03_test[5:, 1], 1), 1)

Results_nofault = np.append(Results_G01, Results_G02,0)
Results_nofault = np.append(Results_nofault, Results_G03, 0)
Results_nofault = np.append(Results_nofault, Results_G04, 0)

np.save("Results_nofault_linear", Results_nofault)


### The above is saved as numpy file

Results_nofault = np.load("Results_nofault_linear.npy")

TimePeriod = [1, 2, 4, 6, 12, 25, 40, 60, 90, 120]
count = 0
num_plots = len(TimePeriod)
#colormap = plt.cm.Accent
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
labels = []
AUC = []
for each in TimePeriod:
    failure_dates = ["2013-10-28", "2013-10-02", "2015-06-15", "2014-08-20", "2016-01-25", "2016-10-24", "2014-01-01", "2016-07-08", "2014-04-07", "2015-12-10", "2017-04-18", "2014-03-01", "2017-04-03", "2015-10-08", "2017-04-03", "2015-01-01", "2014-06-02", "2013-06-01"]
    Model_no = 1
    file = 0
    path = "Data"
    files = [f for f in os.listdir(path)]
    Results = np.zeros([1,2])
    for f in files:
        if not f.startswith('G'):
            COLUMNS = ["Time", "StationID", "MainBearingGTemp", "MainBearingHTemp", "GearOilTemp", "AmbientTemp",
                       "NacelleTemp", "ActivePower", "GenRPM"]
            data = pd.read_csv("Data/" + f, names=COLUMNS, skiprows=1, delimiter=",", decimal=",")
            data.Time = pd.to_datetime(data.Time)

            try:
                X_failure, Y_failure = failure_data(data, n, failure_dates[file], each, True) # deltaT = 90
                sess = tf.Session()
                W = np.load("Weights" +str(Model_no) + ".npy")
                pred_fault = np.dot(X_failure, W)
                Res = np.expand_dims(Y_failure[:,0],1) - pred_fault
                Res = Data_processing.Sliding_window(Res,5)
                Res = np.expand_dims(np.mean(Res,1),1)
                Results_fault = np.append(Res,np.expand_dims(Y_failure[5:,1],1),1)
                Results = np.append(Results, Results_fault,0)
                print("Calculation completed for " + f)
                Model_no = Model_no + 1
                file = file + 1
            except IndexError:
                pass

    Results = np.append(Results_nofault[0:len(Results),:], Results, 0)
    Threshold = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23]
    try:
        fpr, tpr, thresholds = metrics.roc_curve(Results[:,1],Results[:,0], pos_label = 1)
        AUC.append(metrics.roc_auc_score(Results[:,1],Results[:,0]))

        plt.step(fpr, tpr, linewidth = 0.4)
        labels.append("%i days" % each)
    except ValueError:
        pass

plt.legend(labels, ncol=1, loc=4,
           bbox_to_anchor=[1, 0],
           columnspacing=1.0, labelspacing=0.0,
           handletextpad=0.0, handlelength=1.5,
           fancybox=False, shadow=False)

plt.gca().set_title('Linear(' + r'$\mu$' + ')')
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.show()


