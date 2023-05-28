import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import sys

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import keras as keras
from tensorflow.keras import optimizers
np.random.seed(1234)
from keras import backend as K
import os 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from numpy import savetxt, loadtxt
import pickle

import warnings



class multiLSTM(object):
    def __init__(self, file):
        df = pd.read_csv(file, header=None) 
        self.num_stations = len(df.columns) 

        self.inputHorizon = 12 # number of time steps as input
        self.inOutVecDim = self.num_stations  # number of stations
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        self.file_dataset = file 
        with open(self.file_dataset) as f:
            data = csv.reader(f, delimiter=",")
            winds = []
            for line in data:
                winds.append((line))
        self.winds = (np.array(winds)).astype(float) # all data
        self.winds = self.winds[:,:self.inOutVecDim]
        self.means_stds = [0,0]
        self.winds, self.means_stds = self.normalize_winds_0_1(self.winds)
        self.validation_split = 0.05
        self.batchSize = 3
        activation = ['sigmoid',   "tanh",   "relu", 'linear']
        self.activation = activation[2]
        realRun = 0 #1: Will use all data and epochs, 0: Will use 10% of the data and 1 epoch per LSTM
        
        self.epochs, self.dataUsed = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.1]
        self.trainDataRate = 0.8
        self.num_hours = int(df.shape[0]*self.dataUsed)#Trying to parametrize

    def normalize_winds_0_1(self, winds):
        '''normalize based on each station data'''
        stations = winds.shape[1]
        normal_winds = []
        mins_maxs = []
        self.windMax = winds.max()
        self.windMin = winds.min()
        normal_winds = (winds - self.windMin) / self.windMax
        mins_maxs = [self.windMin, self.windMax]
        return np.array(normal_winds), mins_maxs

    def denormalize(self, vec):
        res = vec * self.means_stds[1] + self.means_stds[0] # from 0 to 1
        return res

    def loadData_1(self):
        # for lstm1 output xtrain ytrain
        result = []
        for index in range(len(self.winds) - self.inputHorizon):
            result.append(self.winds[index:index + self.inputHorizon])
        result = np.array(result)  
        trainRow = int(self.num_hours * self.trainDataRate)
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        
        self.xTest = X_train
        self.yTest = y_train
        print('\nNUM HOURS: ', self.num_hours)

        self.predicted = np.zeros_like(self.yTest)

        self.xVal = result[trainRow:self.num_hours-self.inputHorizon, :]
        self.yVal = self.winds[trainRow + self.inputHorizon:self.num_hours]
        self.predictedVal = np.zeros_like(self.yVal)

        return [X_train, y_train]

    def loadData(self, preXTrain, preYTrain, model): # xtrain and ytrain from loadData_1
        # for lstm2 output: xtrain ytrain
        xTrain, yTrain = np.ones_like(preXTrain), np.zeros_like(preYTrain)
  
        for ind in range(len(preXTrain) - self.inputHorizon -1):
            tempInput = preXTrain[ind]
            temp_shape = tempInput.shape
            tempInput = np.reshape(tempInput, (1,temp_shape[0],temp_shape[1]))
            output = model.predict(tempInput)
            tInput = np.reshape(tempInput,temp_shape)
            tempInput = np.vstack((tInput, output))
            tempInput = np.delete(tempInput, 0, axis=0)
            xTrain[ind] = tempInput
            yTrain[ind] = preYTrain[ind+1]
        return [xTrain, yTrain]

  
    def buildModelLSTM_1(self):
        model = Sequential()
        in_nodes = out_nodes = self.inOutVecDim
        layers = [in_nodes, self.num_stations*2, self.num_stations, 32, out_nodes] #changing 57 to num_stations

        model.add(LSTM(units=layers[1], input_dim=layers[0],
            return_sequences=False))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))
    
        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_2(self):
        model = Sequential()
        layers = [self.inOutVecDim, 10 , self.num_stations * 2, 32, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0],
            return_sequences=False))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_3(self):
        model = Sequential()

        layers = [self.inOutVecDim, self.num_stations, self.num_stations * 2, 32, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0], 
            return_sequences=False))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_4(self):
        model = Sequential()

        layers = [self.inOutVecDim, self.num_stations, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0], 
            return_sequences=True))

        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_5(self):
        model = Sequential()

        layers = [self.inOutVecDim, 30, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1], input_dim=layers[0], 
            return_sequences=False))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_6(self):
        model = Sequential()
        layers = [self.inOutVecDim, self.num_stations*2, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1], input_dim=layers[0], 
        return_sequences=True))

        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM(self, lstmModelNum):
        if   lstmModelNum == 1:
            return self.buildModelLSTM_1()
        elif lstmModelNum == 2:
            return self.buildModelLSTM_2()
        elif lstmModelNum == 3:
            return self.buildModelLSTM_3()
        elif lstmModelNum == 4:
            return self.buildModelLSTM_4()
        elif lstmModelNum == 5:
            return self.buildModelLSTM_5()
        elif lstmModelNum == 6:
            return self.buildModelLSTM_6()

    def trainLSTM(self, xTrain, yTrain, lstmModelNum):
        # train first LSTM with inputHorizon number of real input values

        lstmModel = self.buildModelLSTM(lstmModelNum)
        lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, epochs=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
        return lstmModel

    def test(self):
        ''' calculate the predicted values(self.predicted) '''
        for ind in range(len(self.xTest)):
            modelInd = ind % 6 
            
            if modelInd == 0: 
                testInputRaw = self.xTest[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predicted[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])
            
            self.predicted[ind] = self.lstmModels[modelInd].predict(testInput)

        for ind in range(len(self.xVal)):
            modelInd = ind % 6 
            
            if modelInd == 0: 
                testInputRaw = self.xVal[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predictedVal[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])
            
            self.predictedVal[ind] = self.lstmModels[modelInd].predict(testInput)

        return

    def errorMeasures(self, denormalYTest, denormalYPredicted):

        mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
        rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
        nrsme_maxMin = 100*rmse / (denormalYTest.max() - denormalYTest.min())
        nrsme_mean = 100 * rmse / (denormalYTest.mean())

        return mae, rmse, nrsme_maxMin, nrsme_mean

    def drawGraphStation(self, station, visualise = 1, ax = None ):
        '''draw graph of predicted vs real values'''

        yTest = self.yVal[:, station]
        denormalYTest = self.denormalize(yTest)
        np.save('output/test_'+file,denormalYTest)
        denormalPredicted = self.denormalize(self.predictedVal[:, station])
        np.save('output/MLSTM1_'+file,denormalPredicted)
        mae, rmse, nrmse_maxMin, nrmse_mean  = self.errorMeasures(denormalYTest, denormalPredicted)
        print('station %s : MAE = %7.7s   RMSE = %7.7s    nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, rmse, nrmse_maxMin, nrmse_mean ))

        if visualise:
            if ax is None :
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(denormalYTest, label='Real')
            ax.plot(denormalPredicted, label='Predicted', color='red')
            ax.set_xticklabels([0, 100, 200, 300], rotation=40)

        return mae, rmse, nrmse_maxMin, nrmse_mean
        

    def drawGraphAllStations(self):
        if self.num_stations <= 57:
            rows, cols = 5, 10 
        elif self.num_stations >= 100:
            rows, cols = 10, 10 
        maeRmse = np.zeros((rows*cols,4))

        fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
        staInd = 0
        for ax in np.ravel(ax_array): 
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        plt.xticks([0, 100, 200, 300])#, rotation=45)
        errMean = maeRmse.mean(axis=0)
        print(errMean)
        
        return errMean

    def run(self):
        #  training
        xTrain, yTrain = self.loadData_1()
        print(' Training LSTM 1 ...')
        self.lstmModels[0] = self.trainLSTM(xTrain, yTrain, 1)

        for modelInd in range(1,6):
            xTrain, yTrain = self.loadData(xTrain, yTrain, self.lstmModels[modelInd-1])
            print(' Training LSTM %s ...' % (modelInd+1))
            self.lstmModels[modelInd] = self.trainLSTM(xTrain, yTrain, modelInd+1)

        # testing
        print('...... TESTING  ...')
        self.test()

        errMean = self.drawGraphAllStations()
        return errMean

class multiLSTM2(object):
    def __init__(self, file):
        df = pd.read_csv(file, header=None) 
        self.num_stations = len(df.columns) 

        self.inputHorizon = 12 # number of time steps as input
        self.inOutVecDim = self.num_stations  # number of stations
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        self.file_dataset = file 
        with open(self.file_dataset) as f:
            data = csv.reader(f, delimiter=",")
            winds = []
            for line in data:
                winds.append((line))
        self.winds = (np.array(winds)).astype(float) # all data
        self.winds = self.winds[:,:self.inOutVecDim]
        self.means_stds = [0,0]
        self.winds, self.means_stds = self.normalize_winds_0_1(self.winds)
        self.validation_split = 0.05
        self.batchSize = 3
        activation = ['sigmoid',   "tanh",   "relu", 'linear']
        self.activation = activation[1] # MODIFICATION FROM 2 TO 1, RELU TO TANH
        realRun = 0 #1: Will use all data and epochs, 0: Will use 10% of the data and 1 epoch per LSTM
        self.epochs, self.dataUsed = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.1]
        self.trainDataRate = 0.8
        self.num_hours = int(df.shape[0]*self.dataUsed)

    def normalize_winds_0_1(self, winds):
        '''normalize based on each station data'''
        stations = winds.shape[1]
        normal_winds = []
        mins_maxs = []
        windMax = winds.max()
        windMin = winds.min()
        normal_winds = (winds - windMin) / windMax
        mins_maxs = [windMin, windMax]
        return np.array(normal_winds), mins_maxs

    def denormalize(self, vec):
        res = vec * self.means_stds[1] + self.means_stds[0] # from 0 to 1
        return res

    def loadData_1(self):
        # for lstm1 output xtrain ytrain
        result = []
        for index in range(len(self.winds) - self.inputHorizon):
            result.append(self.winds[index:index + self.inputHorizon])
        result = np.array(result)  
        # Lets create three variables for train and test
        trainRow = int(self.num_hours * self.trainDataRate) 
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        
        self.xTest = X_train
        self.yTest = y_train
        print('\nNUM HOURS: ', self.num_hours)

        self.predicted = np.zeros_like(self.yTest)

        self.xVal = result[trainRow:self.num_hours-self.inputHorizon, :]
        self.yVal = self.winds[trainRow + self.inputHorizon:self.num_hours]
        self.predictedVal = np.zeros_like(self.yVal)

        return [X_train, y_train]

    def loadData(self, preXTrain, preYTrain, model): # xtrain and ytrain from loadData_1
        # for lstm2 output: xtrain ytrain
        xTrain, yTrain = np.ones_like(preXTrain), np.zeros_like(preYTrain)
  
        for ind in range(len(preXTrain) - self.inputHorizon -1):
            tempInput = preXTrain[ind]
            temp_shape = tempInput.shape
            tempInput = np.reshape(tempInput, (1,temp_shape[0],temp_shape[1]))
            output = model.predict(tempInput)
            tInput = np.reshape(tempInput,temp_shape)
            tempInput = np.vstack((tInput, output))
            tempInput = np.delete(tempInput, 0, axis=0)
            xTrain[ind] = tempInput
            yTrain[ind] = preYTrain[ind+1]
        return [xTrain, yTrain]

  
    def buildModelLSTM_1(self):
        model = Sequential()
        in_nodes = out_nodes = self.inOutVecDim
        layers = [in_nodes, self.num_stations*2, self.num_stations, 32, out_nodes] #changing 57 to num_stations

        model.add(LSTM(units=layers[1], input_dim=layers[0],
            return_sequences=False))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))
    
        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_2(self):
        model = Sequential()
        layers = [self.inOutVecDim, 10 , self.num_stations * 2, 32, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0],
            return_sequences=False))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_3(self):
        model = Sequential()

        layers = [self.inOutVecDim, self.num_stations, self.num_stations * 2, 32, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0], 
            return_sequences=False))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_4(self):
        model = Sequential()

        layers = [self.inOutVecDim, self.num_stations, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0], 
            return_sequences=True))

        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_5(self):
        model = Sequential()

        layers = [self.inOutVecDim, 30, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1], input_dim=layers[0], 
            return_sequences=False))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_6(self):
        model = Sequential()
        layers = [self.inOutVecDim, self.num_stations*2, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1], input_dim=layers[0], 
        return_sequences=True))


        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM(self, lstmModelNum):
        if   lstmModelNum == 1:
            return self.buildModelLSTM_1()
        elif lstmModelNum == 2:
            return self.buildModelLSTM_2()
        elif lstmModelNum == 3:
            return self.buildModelLSTM_3()
        elif lstmModelNum == 4:
            return self.buildModelLSTM_4()
        elif lstmModelNum == 5:
            return self.buildModelLSTM_5()
        elif lstmModelNum == 6:
            return self.buildModelLSTM_6()

    def trainLSTM(self, xTrain, yTrain, lstmModelNum):
        # train first LSTM with inputHorizon number of real input values

        lstmModel = self.buildModelLSTM(lstmModelNum)
        lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, epochs=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
        return lstmModel

    def test(self):
        ''' calculate the predicted values(self.predicted) '''
        for ind in range(len(self.xTest)):
            
            modelInd = ind % 6 
            
            if modelInd == 0: 
                testInputRaw = self.xTest[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predicted[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])
            
            self.predicted[ind] = self.lstmModels[modelInd].predict(testInput)

        for ind in range(len(self.xVal)):
            
            modelInd = ind % 6 
            
            if modelInd == 0: 
                testInputRaw = self.xVal[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predictedVal[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])
            
            self.predictedVal[ind] = self.lstmModels[modelInd].predict(testInput)

        return

    def errorMeasures(self, denormalYTest, denormalYPredicted):

        mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
        rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
        nrsme_maxMin = 100*rmse / (denormalYTest.max() - denormalYTest.min())
        nrsme_mean = 100 * rmse / (denormalYTest.mean())

        return mae, rmse, nrsme_maxMin, nrsme_mean

    def drawGraphStation(self, station, visualise = 1, ax = None ):
        '''draw graph of predicted vs real values'''

        yTest = self.yVal[:, station]
        denormalYTest = self.denormalize(yTest)

        denormalPredicted = self.denormalize(self.predictedVal[:, station])
        np.save('output/MLSTM2_'+file,denormalPredicted)
        mae, rmse, nrmse_maxMin, nrmse_mean  = self.errorMeasures(denormalYTest, denormalPredicted)
        print('station %s : MAE = %7.7s   RMSE = %7.7s    nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, rmse, nrmse_maxMin, nrmse_mean ))

        if visualise:
            if ax is None :
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(denormalYTest, label='Real')
            ax.plot(denormalPredicted, label='Predicted', color='red')
            ax.set_xticklabels([0, 100, 200, 300], rotation=40)

        return mae, rmse, nrmse_maxMin, nrmse_mean

    def drawGraphAllStations(self):
        if self.num_stations <= 57:
            rows, cols = 5, 10 
        elif self.num_stations >= 100:
            rows, cols = 10, 10 
        maeRmse = np.zeros((rows*cols,4))

        fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
        staInd = 0
        for ax in np.ravel(ax_array): 
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        plt.xticks([0, 100, 200, 300])#, rotation=45)
        errMean = maeRmse.mean(axis=0)
        print(errMean)
        
        return errMean

    def run(self):
        #  training
        xTrain, yTrain = self.loadData_1()
        print(' Training LSTM 1 ...')
        self.lstmModels[0] = self.trainLSTM(xTrain, yTrain, 1)

        for modelInd in range(1,6):
            xTrain, yTrain = self.loadData(xTrain, yTrain, self.lstmModels[modelInd-1])
            print(' Training LSTM %s ...' % (modelInd+1))
            self.lstmModels[modelInd] = self.trainLSTM(xTrain, yTrain, modelInd+1)

        # testing
        print('...... TESTING  ...')
        self.test()

        errMean = self.drawGraphAllStations()
       
        return errMean

class multiLSTM3(object):
    def __init__(self, file):
        df = pd.read_csv(file, header=None) 
        self.num_stations = len(df.columns)

        self.inputHorizon = 12 # number of time steps as input
        self.inOutVecDim = self.num_stations  # number of stations
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        self.file_dataset = file 
        with open(self.file_dataset) as f:
            data = csv.reader(f, delimiter=",")
            winds = []
            for line in data:
                winds.append((line))
        self.winds = (np.array(winds)).astype(float) # all data
        self.winds = self.winds[:,:self.inOutVecDim]
        self.means_stds = [0,0]
        self.winds, self.means_stds = self.normalize_winds_0_1(self.winds)
        self.validation_split = 0.05
        self.batchSize = 3
        activation = ['sigmoid',   "tanh",   "relu", 'linear']
        self.activation = activation[2]
        realRun = 0 #1: Will use all data and epochs, 0: Will use 10% of the data and 1 epoch per LSTM
        self.epochs, self.dataUsed = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.1]
        self.trainDataRate = 0.8
        self.num_hours = int(df.shape[0]*self.dataUsed)
    def normalize_winds_0_1(self, winds):
        '''normalize based on each station data'''
        stations = winds.shape[1]
        normal_winds = []
        mins_maxs = []
        windMax = winds.max()
        windMin = winds.min()
        normal_winds = (winds - windMin) / windMax
        mins_maxs = [windMin, windMax]
        return np.array(normal_winds), mins_maxs

    def denormalize(self, vec):
        res = vec * self.means_stds[1] + self.means_stds[0] # from 0 to 1
        return res

    def loadData_1(self):
        # for lstm1 output xtrain ytrain
        result = []
        for index in range(len(self.winds) - self.inputHorizon):
            result.append(self.winds[index:index + self.inputHorizon])
        result = np.array(result)  
        trainRow = int(self.num_hours * self.trainDataRate) 
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        
        self.xTest = X_train
        self.yTest = y_train
        print('\nNUM HOURS: ', self.num_hours)

        self.predicted = np.zeros_like(self.yTest)

        self.xVal = result[trainRow:self.num_hours-self.inputHorizon, :]
        self.yVal = self.winds[trainRow + self.inputHorizon:self.num_hours]
        self.predictedVal = np.zeros_like(self.yVal)

        return [X_train, y_train]

    def loadData(self, preXTrain, preYTrain, model): # xtrain and ytrain from loadData_1
        # for lstm2 output: xtrain ytrain
        xTrain, yTrain = np.ones_like(preXTrain), np.zeros_like(preYTrain)
  
        for ind in range(len(preXTrain) - self.inputHorizon -1):
            tempInput = preXTrain[ind]
            temp_shape = tempInput.shape
            tempInput = np.reshape(tempInput, (1,temp_shape[0],temp_shape[1]))
            output = model.predict(tempInput)
            tInput = np.reshape(tempInput,temp_shape)
            tempInput = np.vstack((tInput, output))
            tempInput = np.delete(tempInput, 0, axis=0)
            xTrain[ind] = tempInput
            yTrain[ind] = preYTrain[ind+1]
        return [xTrain, yTrain]

  
    def buildModelLSTM_1(self):
        model = Sequential()
        in_nodes = out_nodes = self.inOutVecDim
        layers = [in_nodes, self.num_stations*2, self.num_stations, 32, out_nodes] #changing 57 to num_stations
        
        model.add(LSTM(units=layers[1], input_dim=layers[0],
            return_sequences=False, dropout=0.2))
        
        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))
    
        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_2(self):
        model = Sequential()
        layers = [self.inOutVecDim, 10 , self.num_stations * 2, 32, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0],
            return_sequences=False, dropout=0.2))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_3(self):
        model = Sequential()

        layers = [self.inOutVecDim, self.num_stations, self.num_stations * 2, 32, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0], 
            return_sequences=False, dropout=0.2))

        model.add(Dense(
            units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_4(self):
        model = Sequential()

        layers = [self.inOutVecDim, self.num_stations, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1],input_dim=layers[0], 
            return_sequences=True))

        model.add(LSTM(layers[2],
            return_sequences=False, dropout=0.2))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_5(self):
        model = Sequential()

        layers = [self.inOutVecDim, 30, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1], input_dim=layers[0], 
            return_sequences=False, dropout=0.2))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM_6(self):
        model = Sequential()
        layers = [self.inOutVecDim, self.num_stations*2, self.num_stations * 2, self.num_stations, self.inOutVecDim]
        model.add(LSTM(units=layers[1], input_dim=layers[0], 
        return_sequences=True, dropout=0.2))


        model.add(LSTM(layers[2],
            return_sequences=False))

        model.add(Dense(units=layers[4]))

        model.add(Activation(self.activation))

        optimizer = optimizers.RMSprop(lr=0.001)
        model.compile(loss="mae", optimizer=optimizer)

        return model

    def buildModelLSTM(self, lstmModelNum):
        if   lstmModelNum == 1:
            return self.buildModelLSTM_1()
        elif lstmModelNum == 2:
            return self.buildModelLSTM_2()
        elif lstmModelNum == 3:
            return self.buildModelLSTM_3()
        elif lstmModelNum == 4:
            return self.buildModelLSTM_4()
        elif lstmModelNum == 5:
            return self.buildModelLSTM_5()
        elif lstmModelNum == 6:
            return self.buildModelLSTM_6()

    def trainLSTM(self, xTrain, yTrain, lstmModelNum):

        lstmModel = self.buildModelLSTM(lstmModelNum)
        lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, epochs=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
        return lstmModel

    def test(self):
        ''' calculate the predicted values(self.predicted) '''
        for ind in range(len(self.xTest)):
            
            modelInd = ind % 6 
            
            if modelInd == 0: 
                testInputRaw = self.xTest[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predicted[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])
            
            self.predicted[ind] = self.lstmModels[modelInd].predict(testInput)

        for ind in range(len(self.xVal)):
            
            modelInd = ind % 6 
            
            if modelInd == 0: 
                testInputRaw = self.xVal[ind]
                testInputShape = testInputRaw.shape
                testInput = np.reshape(testInputRaw, [1, testInputShape[0], testInputShape[1]])
            else :
                testInputRaw = np.vstack((testInputRaw, self.predictedVal[ind-1]))
                testInput = np.delete(testInputRaw, 0, axis=0)
                testInputShape = testInput.shape
                testInput = np.reshape(testInput, [1, testInputShape[0], testInputShape[1]])
            
            self.predictedVal[ind] = self.lstmModels[modelInd].predict(testInput)

        return

    def errorMeasures(self, denormalYTest, denormalYPredicted):

        mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
        rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
        nrsme_maxMin = 100*rmse / (denormalYTest.max() - denormalYTest.min())
        nrsme_mean = 100 * rmse / (denormalYTest.mean())

        return mae, rmse, nrsme_maxMin, nrsme_mean

    def drawGraphStation(self, station, visualise = 1, ax = None ):
        '''draw graph of predicted vs real values'''

        yTest = self.yVal[:, station]
        denormalYTest = self.denormalize(yTest)

        denormalPredicted = self.denormalize(self.predictedVal[:, station])
        np.save('output/MLSTM3_'+file,denormalPredicted)
        mae, rmse, nrmse_maxMin, nrmse_mean  = self.errorMeasures(denormalYTest, denormalPredicted)
        print('station %s : MAE = %7.7s   RMSE = %7.7s    nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, rmse, nrmse_maxMin, nrmse_mean ))

        if visualise:
            if ax is None :
                fig = plt.figure()
                ax = fig.add_subplot(111)

            ax.plot(denormalYTest, label='Real')
            ax.plot(denormalPredicted, label='Predicted', color='red')
            ax.set_xticklabels([0, 100, 200, 300], rotation=40)

        return mae, rmse, nrmse_maxMin, nrmse_mean

    def drawGraphAllStations(self):
        if self.num_stations <= 57:
            rows, cols = 5, 10 
        elif self.num_stations >= 100:
            rows, cols = 10, 10 
        maeRmse = np.zeros((rows*cols,4))

        fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
        staInd = 0
        for ax in np.ravel(ax_array): 
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        plt.xticks([0, 100, 200, 300])#, rotation=45)
        errMean = maeRmse.mean(axis=0)
        print(errMean)

        return errMean

    def run(self):
        #  training
        xTrain, yTrain = self.loadData_1()
        print(' Training LSTM 1 ...')
        self.lstmModels[0] = self.trainLSTM(xTrain, yTrain, 1)

        for modelInd in range(1,6):
            xTrain, yTrain = self.loadData(xTrain, yTrain, self.lstmModels[modelInd-1])
            print(' Training LSTM %s ...' % (modelInd+1))
            self.lstmModels[modelInd] = self.trainLSTM(xTrain, yTrain, modelInd+1)

        # testing
        print('...... TESTING  ...')
        self.test()

        errMean = self.drawGraphAllStations()
        return errMean

# load models from file
def load_all_models(file):
    print('\nMultiLSTM 1\n')
    DeepForecast1 = multiLSTM(file)
    results1 = DeepForecast1.run()

    print('\nMultiLSTM 2\n')
    DeepForecast2 = multiLSTM2(file)
    results2 = DeepForecast2.run()

    print('\nMultiLSTM 3\n')
    DeepForecast3 = multiLSTM3(file)
    results3 = DeepForecast3.run()
    
    all_models = [DeepForecast1, DeepForecast2, DeepForecast3]
    return all_models, [results1, results2, results3]

def stacked_dataset(members):
    models_predictions_train = np.empty((len(members[0].yTest), members[0].num_stations, len(members)), float) #np.zeros((self.xTest.shape, len(members)))
    models_predictions_val = np.empty((len(members[0].yVal), members[0].num_stations, len(members)), float) #np.zeros((self.xTest.shape, len(members)))

    num_model = 0
    for model in members:
        model_prediction_train = np.empty((len(members[0].yTest), model.num_stations), float)
        model_prediction_val = np.empty((len(members[0].yVal), model.num_stations), float)

        for station in range(model.num_stations):
            station_prediction_train = model.denormalize(model.predicted[:, station])
            station_prediction_val = model.denormalize(model.predictedVal[:, station])
            model_prediction_train[:, station] = station_prediction_train
            model_prediction_val[:, station] = station_prediction_val

        models_predictions_train[:, :, num_model] = model_prediction_train
        models_predictions_val[:, :, num_model] = model_prediction_val #THIS SEEMS TO NOR WORK
        num_model+=1

    models_predictions_train_norm = (models_predictions_train - members[0].windMin) / members[0].windMax
    models_predictions_val_norm = (models_predictions_val - members[0].windMin) / members[0].windMax
    return models_predictions_train_norm, models_predictions_val_norm
    
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members):
    from sklearn.linear_model import LinearRegression
    # create dataset using ensemble
    models_predictions_train_norm, models_predictions_val_norm = stacked_dataset(members)
    yTest = members[0].yTest
    
    models = []
    print('\n=========================================================================================================')
    print('\nTraining Linear Regression models (one per station)...\n')
    print('=========================================================================================================\n')
    for station in range(members[0].num_stations):
        model = LinearRegression()
        model.fit(models_predictions_train_norm[:,station,:], yTest[:,station])
        models.append(model)
    return models

# make a prediction with the stacked model
def stacked_prediction(members, models):
    # create dataset using ensemble
    models_predictions_train_norm, models_predictions_val_norm = stacked_dataset(members)
    y_stacked = np.zeros(shape=(members[0].yVal.shape))
    # make a prediction
    for station in range(members[0].num_stations):
        y_stacked[:,station] = models[station].predict(models_predictions_val_norm[:,station,:])
    return y_stacked

def errorMeasures_stacked(denormalYTest, denormalYPredicted):

    mae = np.mean(np.absolute(denormalYTest - denormalYPredicted))
    rmse = np.sqrt((np.mean((np.absolute(denormalYTest - denormalYPredicted)) ** 2)))
    nrsme_maxMin = 100*rmse / (denormalYTest.max() - denormalYTest.min())
    nrsme_mean = 100 * rmse / (denormalYTest.mean())

    return mae, rmse, nrsme_maxMin, nrsme_mean

def drawGraphStation_stacked(members, yPred, yTest, station, visualise = 1, ax = None ):
    '''draw graph of predicted vs real values'''
    member0 = members[0]
    yTest = yTest[:, station]
    denormalYTest = member0.denormalize(yTest)
    denormalPredicted = member0.denormalize(yPred[:, station])
    #np.save('output/stacked_'+file,denormalPredicted)
    mae, rmse, nrmse_maxMin, nrmse_mean  = errorMeasures_stacked(denormalYTest, denormalPredicted)
    print('station %s : MAE = %7.7s   RMSE = %7.7s    nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, rmse, nrmse_maxMin, nrmse_mean ))

    if visualise:
        if ax is None :
            fig = plt.figure()
            ax = fig.add_subplot(111)

        ax.plot(denormalYTest, label='Real')
        ax.plot(denormalPredicted, label='Predicted', color='red')
        ax.set_xticklabels([0, 100, 200, 300], rotation=40)

    return mae, rmse, nrmse_maxMin, nrmse_mean

def drawGraphAllStations_stacked(members, yPred, yTest, num_stations):
    
    if num_stations <= 57:
        rows, cols = 5, 10 
    elif num_stations >= 100:
        rows, cols = 10, 10 
    maeRmse = np.zeros((rows*cols,4))

    fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
    staInd = 0
    for ax in np.ravel(ax_array): 
        maeRmse[staInd] = drawGraphStation_stacked(members, yPred, yTest, staInd, visualise=1, ax=ax)
        staInd += 1
    plt.xticks([0, 100, 200, 300])#, rotation=45)
    errMean = maeRmse.mean(axis=0)
    print(errMean)
    
    return errMean

original = False
if original:
    file = 'MS_winds.mat'
    
    members, results = load_all_models('data/'+file)
    stacked_models = fit_stacked_model(members)
    yPred = stacked_prediction(members, stacked_models)
    print('\n STACKED MODEL RESULTS \n')
    errMean = drawGraphAllStations_stacked(members, yPred, members[0].yVal, members[0].num_stations)
    print('\n')
    
else:
    for i in range(1):
        print('\n')
        print('===============================================================================================================================')
        print('ITERATION ',i)
        print('===============================================================================================================================')
        print('\n')
        contador = 1
        ids, mae, rmse, nrmse_maxMin, nrmse_mean = [], [], [], [], []
        for file in os.listdir('data'):
            print('\n')
            print('===============================================================================================================================')
            print('Experiment number: ',contador,'/ 31')
            print('Dataset: ',file)
            print('===============================================================================================================================')
            print('\n')
            members, results = load_all_models('data/'+file)
            stacked_model = fit_stacked_model(members)
            yPred = stacked_prediction(members, stacked_model)

            errMean = drawGraphAllStations_stacked(members, yPred, members[0].yVal, members[0].num_stations)

            contador+=1
