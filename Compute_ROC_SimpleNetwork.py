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

def SimpleNetwork(X_train, Y_train, Model):
    tf.reset_default_graph()
    sess = tf.Session()
    # First let's load meta graph and restore weights
    saver = tf.train.import_meta_graph(Model + '.meta')
    saver.restore(sess, Model)

    graph = tf.get_default_graph()
    W_out = graph.get_tensor_by_name("Output_weights:0")
    W_hid = graph.get_tensor_by_name("Hidden_weights:0")

    def multilayer_perceptron(x, W_hidden, W_output):
        # Hidden layer with RELU activation
        hidden_layer = tf.matmul(x, W_hidden)
        hidden_layer = tf.sigmoid(hidden_layer)
        # Output layer with linear activation
        out_layer = tf.matmul(hidden_layer, W_output)
        return out_layer

    prediction = multilayer_perceptron(tf.cast(X_train, tf.float32), W_hid, W_out)
    residuals = np.expand_dims(Y_train[:,0],1) - sess.run(prediction)

    return sess.run(prediction), residuals

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

X_G01, Y_G01 = no_failure_data(data1, n, 20)
X_G02, Y_G02 = no_failure_data(data2, n, 20)
X_G03, Y_G03 = no_failure_data(data3, n, 20)
X_G04, Y_G04 = no_failure_data(data4, n, 20)

sess = tf.Session()
pred_G01, Res_G01 = SimpleNetwork(X_G01, Y_G01, 'SimpleNetwork-40')
pred_G02, Res_G02 = SimpleNetwork(X_G02, Y_G02, 'SimpleNetwork-41')
pred_G03, Res_G03 = SimpleNetwork(X_G03, Y_G03, 'SimpleNetwork-42')
pred_G04, Res_G04 = SimpleNetwork(X_G04, Y_G04, 'SimpleNetwork-43')

Res_G01 = Data_processing.Sliding_window(Res_G01, 5)
Res_G01 = np.expand_dims(np.mean(Res_G01, 1), 1)
Res_G02 = Data_processing.Sliding_window(Res_G02, 5)
Res_G02 = np.expand_dims(np.mean(Res_G02, 1), 1)
Res_G03 = Data_processing.Sliding_window(Res_G03, 5)
Res_G03 = np.expand_dims(np.mean(Res_G03, 1), 1)
Res_G04 = Data_processing.Sliding_window(Res_G04, 5)
Res_G04 = np.expand_dims(np.mean(Res_G04, 1), 1)

Results_G01 = np.append(Res_G01, np.expand_dims(Y_G01[5:, 1], 1), 1)
Results_G02 = np.append(Res_G02, np.expand_dims(Y_G02[5:, 1], 1), 1)
Results_G03 = np.append(Res_G03, np.expand_dims(Y_G03[5:, 1], 1), 1)
Results_G04 = np.append(Res_G04, np.expand_dims(Y_G04[5:, 1], 1), 1)

Results_nofault = np.append(Results_G01, Results_G02,0)
Results_nofault = np.append(Results_nofault, Results_G03, 0)
Results_nofault = np.append(Results_nofault, Results_G04, 0)

np.save("Results_nofault", Results_nofault)


### The above is saved as numpy file

Results_nofault = np.load("Results_nofault.npy")

TimePeriod = [1, 2, 4, 6, 12, 25, 40, 60, 90, 120]
count = 0
num_plots = len(TimePeriod)
#colormap = plt.cm.Accent
#plt.gca().set_color_cycle([colormap(i) for i in np.linspace(0, 0.9, num_plots)])
labels = []
AUC = []
for each in TimePeriod:
    failure_dates = ["2013-10-28", "2013-10-02", "2015-06-15", "2014-08-20", "2016-01-25", "2016-10-24", "2014-01-01", "2016-07-08", "2014-04-07", "2015-12-10", "2017-04-18", "2014-03-01", "2017-04-03", "2015-10-08", "2017-04-03", "2015-01-01", "2014-06-02", "2013-06-01"]
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
                pred_fault, Res = SimpleNetwork(X_failure, Y_failure, 'SimpleNetwork-' + str(file + 1))
                Res = Data_processing.Sliding_window(Res,10)
                Res = np.expand_dims(np.mean(Res,1),1)
                Results_fault = np.append(Res,np.expand_dims(Y_failure[10:,1],1),1)
                Results = np.append(Results, Results_fault,0)
                print("Calculation completed for " + f)
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
#plt.title("ANN(r'$mu$') model")
plt.gca().set_title('ANN(' + r'$\mu$' + ')')
plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.show()


