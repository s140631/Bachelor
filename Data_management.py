import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from datetime import datetime, timedelta
import pandas as pd
import random
from itertools import islice
import os

## THESE METHODS HAVE BEEN USED TO IMPORT AND MANIPULATE THE DATA

class Data_processing:

    def failure_data(data, n, failure_date):
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
            data['labels'].loc[data.Time > failure_date] = 1


            training_data = data
            start = training_data.Time[0] + timedelta(days = 60)
            training_data = training_data.drop(training_data[training_data.Time < start].index)  # Delete the first months
            training_data = training_data.reset_index()
            del training_data['index']

            ### Filters!

            X_test = training_data[["GearOilTemp", "AmbientTemp", "NacelleTemp", "ActivePower", "GenRPM"]]
            Y_test = training_data[["MainBearingHTemp", "labels"]]
            X_test = np.asarray( X_test)
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

    def test_data(data, n, year):
            length_of_window = n
            data = data[~pd.isnull(data.StationID)]
            data = data[~pd.isnull(data.MainBearingGTemp)]
            data = data[~pd.isnull(data.MainBearingHTemp)]
            data = data[~pd.isnull(data.GearOilTemp)]
            data = data[~pd.isnull(data.AmbientTemp)]
            data = data[~pd.isnull(data.NacelleTemp)]
            data = data[~pd.isnull(data.ActivePower)]
            data = data[~pd.isnull(data.GenRPM)]

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


            if data.Time[0].month + 2 > 12:
                start_month = data.Time[0].month - 10
                start_year = data.Time[0].year + 1
            else:
                start_month = data.Time[0].month + 2
                start_year = data.Time[0].year

            if start_month + 2 > 12:
                end_month = start_month - 10
                end_year = start_year + 2
            else:
                end_month = start_month + 2
                end_year = start_year + 1

            start_month = end_month
            start_year = end_year + (year - 2) # Add 1 year if we wants to extract year 3
            end_year = start_year + 1

            training_data = data
            training_data = training_data.drop(data[data.Time < str(start_year) + '-' + str(
                start_month) + '-01 00:00:00'].index)  # Delete the first 2 months
            training_data = training_data.drop(training_data[training_data.Time > str(end_year) + '-' + str(
                end_month) + '-01 00:00:00'].index)  # Extract 14 months of data, assuming no-fault state
            training_data = training_data.reset_index()
            del training_data['index']

            print("Starting at date " + str(start_year) + str(start_month) + '-01 00:00:00')
            ### Outliers!

            def mad_based_outlier(points, thresh=3.5):
                if len(points.shape) == 1:
                    points = points[:, None]
                median = np.median(points, axis=0)
                diff = np.sum((points - median) ** 2, axis=-1)
                diff = np.sqrt(diff)
                med_abs_deviation = np.median(diff)

                modified_z_score = 0.6745 * diff / med_abs_deviation

                return modified_z_score > thresh

            training_data = training_data.drop(training_data[mad_based_outlier(training_data.MainBearingHTemp)].index)
            training_data = training_data.reset_index()


            X_test = training_data[["GearOilTemp", "AmbientTemp", "NacelleTemp", "ActivePower", "GenRPM"]]
            Y_test = training_data[["MainBearingHTemp"]]
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

    def define_data(data, n):
            length_of_window = n
            data = data[~pd.isnull(data.StationID)]
            data = data[~pd.isnull(data.MainBearingGTemp)]
            data = data[~pd.isnull(data.MainBearingHTemp)]
            data = data[~pd.isnull(data.GearOilTemp)]
            data = data[~pd.isnull(data.AmbientTemp)]
            data = data[~pd.isnull(data.NacelleTemp)]
            data = data[~pd.isnull(data.ActivePower)]
            data = data[~pd.isnull(data.GenRPM)]


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


            if data.Time[0].month + 2 > 12:
                start_month = data.Time[0].month - 10
                start_year = data.Time[0].year + 1
            else:
                start_month = data.Time[0].month + 2
                start_year = data.Time[0].year

            if start_month + 2 > 12:
                end_month = start_month - 10
                end_year = start_year + 2
            else:
                end_month = start_month + 2
                end_year = start_year + 1

            training_data = data
            training_data = training_data.drop(data[data.Time < str(start_year) + '-' + str(
                start_month) + '-01 00:00:00'].index)  # Delete the first 2 months
            training_data = training_data.drop(training_data[training_data.Time > str(end_year) + '-' + str(
                end_month) + '-01 00:00:00'].index)  # Extract 14 months of data, assuming no-fault state
            training_data = training_data.reset_index()
            del training_data['index']

            ### Outliers!

            def mad_based_outlier(points, thresh=3.5):
                if len(points.shape) == 1:
                    points = points[:, None]
                median = np.median(points, axis=0)
                diff = np.sum((points - median) ** 2, axis=-1)
                diff = np.sqrt(diff)
                med_abs_deviation = np.median(diff)

                modified_z_score = 0.6745 * diff / med_abs_deviation

                return modified_z_score > thresh

            training_data = training_data.drop(training_data[mad_based_outlier(training_data.MainBearingHTemp)].index)
            training_data = training_data.reset_index()
            ## Validation set - 20 random sequences within 72 hours
            validation_data = training_data[0:0]
            for i in range(0, 20):
                rnd = random.randint(1, (len(training_data.Time)-20))
                val_temp = training_data
                end_date = val_temp.Time[rnd] + timedelta(hours=72)
                val_temp = val_temp[val_temp.Time > val_temp.Time[rnd]]
                val_temp = val_temp[val_temp.Time < end_date]
                validation_data = validation_data.append(val_temp)
            validation_index = validation_data.index


            ## Test set - 20 random sequences within 72 hours
            test_data = training_data[0:0]
            for i in range(0, 20):
                rnd = random.randint(0, len(training_data.Time)-20)
                test_temp = training_data
                end_date = test_temp.Time[rnd] + timedelta(hours=72)
                test_temp = test_temp[test_temp.Time > test_temp.Time[rnd]]
                test_temp = test_temp[test_temp.Time < end_date]
                test_data = test_data.append(test_temp)
                test_index = test_data.index
                test_index = test_index

            X_train = training_data[["GearOilTemp", "AmbientTemp", "NacelleTemp", "ActivePower", "GenRPM"]]
            Y_train = training_data[["MainBearingHTemp"]]
            X_train = np.asarray(X_train)
            Y_train = np.asarray(Y_train)


            ##### Establish sliding window!
            n = length_of_window
            X_train = Data_processing.Sliding_window(X_train, n)
            Y_train = np.delete(Y_train, [range(n)], 0)
            #X_test = Data_processing.Sliding_window(X_test, n)
            #Y_test = np.delete(Y_test, [range(n)], 0)
            #X_val = Data_processing.Sliding_window(X_val, n)
            #Y_val = np.delete(Y_val, [range(n)], 0)
            test_index = test_index[test_index < X_train.shape[0]]
            validation_index = validation_index[validation_index < X_train.shape[0]]

            X_test = X_train[test_index, :]
            Y_test = Y_train[test_index, :]
            X_val = X_train[validation_index, :]
            Y_val = Y_train[validation_index, :]

            # Adding bias term to X
            X_BIAS = np.ones((X_train.shape[0], X_train.shape[1] + 1))
            X_BIAS[:, :-1] = X_train
            X_train = X_BIAS
            X_BIAS = np.ones((X_test.shape[0], X_test.shape[1] + 1))
            X_BIAS[:, :-1] = X_test
            X_test = X_BIAS
            X_BIAS = np.ones((X_val.shape[0], X_val.shape[1] + 1))
            X_BIAS[:, :-1] = X_val
            X_val = X_BIAS

            # Test and validation set


            return X_train, Y_train, X_test, Y_test, X_val, Y_val

    def load_data(path, length_of_window):

            COLUMNS = ["Time", "StationID", "MainBearingGTemp", "MainBearingHTemp", "GearOilTemp", "AmbientTemp",
                       "NacelleTemp", "ActivePower", "GenRPM"]
            files = [f for f in os.listdir(path)]
            first_run = True
            for f in files:
                data = pd.read_csv("Data/" + f, names=COLUMNS, skiprows=1, delimiter=";", decimal=",")
                # data.Time = str(data.Time)
                data.Time = pd.to_datetime(data.Time)
                print("Loading " + str(f))
                if first_run:
                    X_train, Y_train, X_test, Y_test, X_val, Y_val = Data_processing.define_data(data, length_of_window)
                    first_run = False
                else:
                    X_train_t, Y_train_t, X_test_t, Y_test_t, X_val_t, Y_val_t = Data_processing.define_data(data,length_of_window)
                    X_train = np.append(X_train, X_train_t, 0)
                    Y_train = np.append(Y_train, Y_train_t, 0)
                    X_test = np.append(X_test, X_test_t, 0)
                    Y_test = np.append(Y_test, Y_test_t, 0)
                    X_val = np.append(X_val, X_val_t, 0)
                    Y_val = np.append(Y_val, Y_val_t, 0)

            return X_train, Y_train, X_test, Y_test, X_val, Y_val

    def Sliding_window(data, n):

            it = iter(data)
            window_full = np.append(data[0, :], data[0, :])
            for i in range(1, n - 1):
                window_full = np.append(data[i, :], window_full)

            for i in range(n, len(data) - 1):
                window = np.append(data[i + 1, :], data[i, :])
                for j in range(1, n - 1):
                    window = np.append(window, data[i - j, :])
                window_full = np.vstack((window_full, window))
            return window_full
