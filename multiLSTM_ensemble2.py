import matplotlib.pyplot as plt
import numpy as np
import time
import csv
import sys
import tensorflow as tf
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout, Flatten
from keras.layers import Convolution2D
from keras.layers.recurrent import LSTM, SimpleRNN, GRU
import keras as keras
from tensorflow.keras import optimizers
np.random.seed(1234)
from keras import backend as K
#For our experiments:
import os 
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from numpy import savetxt, loadtxt
import pickle


class multiLSTM(object):
    def __init__(self, file):
        df = pd.read_csv(file, header=None) #Trying to parametrize
        self.num_stations = len(df.columns) #Trying to parametrize
        #print('Testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',self.num_hours, self.num_stations)

        self.inputHorizon = 12 # number of time steps as input
        self.inOutVecDim = self.num_stations  # number of stations: Need to change depending on data!!!
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        #file_dataset = 'C:/Users/Jaime/Desktop/Minerva/workspace/nrel/Deep-Forecast/MS_winds.dat'
        self.file_dataset = file #trying
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
        realRun = 1 #Change from 0 to 1 if you want to use a 0.5% of the data to train
        #          model number :           1   2   3   4   5   6
        self.epochs, self.dataUsed = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.1]# percentage of data used for training(saving time for debuging)
        self.trainDataRate = 0.8
        self.num_hours = int(df.shape[0]*self.dataUsed)#Trying to parametrize

        #print('Testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',self.epochs, self.trainDataRate) #I have changed from 1 to 0.715 since 6000 out of 
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
        # Lets create three variables for train and test
        trainRow = int(self.num_hours * self.trainDataRate) #Here we have the errors, datasets ad-hoc !!! Lets change it, instead of 6000 lets use percentages
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        # self.xTest = result[6000:6999, :]
        # self.yTest = self.winds[6000 + self.inputHorizon:6999 + self.inputHorizon]
        # testRow = int(self.num_hours * self.trainDataRate*2)
        self.xTest = X_train
        self.yTest = y_train
        print('\nNUM HOURS: ', self.num_hours)

        # self.xTest = result[trainRow:testRow, :]
        # self.yTest = self.winds[trainRow + self.inputHorizon:testRow+ self.inputHorizon]
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

        # model.add(LSTM(input_dim=layers[0],output_dim=layers[1],
        #     return_sequences=False))
        model.add(LSTM(units=layers[1], input_dim=layers[0],
            return_sequences=False))

        # model.add(Dense(
        #     output_dim=layers[4]))
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
        # lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, nb_epoch=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
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

    # def drawGraphStation(self, station, visualise = 1, ax = None ):
    #     '''draw graph of predicted vs real values'''

    #     yTest = self.yTest[:, station]
    #     denormalYTest = self.denormalize(yTest)

    #     denormalPredicted = self.denormalize(self.predicted[:, station])
    #     mae, rmse, nrmse_maxMin, nrmse_mean  = self.errorMeasures(denormalYTest, denormalPredicted)
    #     print('station %s : MAE = %7.7s   RMSE = %7.7s    nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, rmse, nrmse_maxMin, nrmse_mean ))

    #     if visualise:
    #         if ax is None :
    #             fig = plt.figure()
    #             ax = fig.add_subplot(111)

    #         ax.plot(denormalYTest, label='Real')
    #         ax.plot(denormalPredicted, label='Predicted', color='red')
    #         ax.set_xticklabels([0, 100, 200, 300], rotation=40)

    #     return mae, rmse, nrmse_maxMin, nrmse_mean

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
            rows, cols = 5, 10 # rows*cols < number of stations !!!
        elif self.num_stations >= 100:
            rows, cols = 10, 10 # rows*cols < number of stations !!!
        maeRmse = np.zeros((rows*cols,4))

        fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
        staInd = 0
        for ax in np.ravel(ax_array): 
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        plt.xticks([0, 100, 200, 300])#, rotation=45)
        errMean = maeRmse.mean(axis=0)
        print(errMean)
        #filename = 'pgf/finalEpoch'
        #plt.savefig('{}.pgf'.format(filename)) #ERROR
        #plt.savefig('{}.pdf'.format(filename))
        #plt.savefig('output/'+file.replace('.mat', '.png'))
        
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
        df = pd.read_csv(file, header=None) #Trying to parametrize
        self.num_stations = len(df.columns) #Trying to parametrize
        #print('Testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',self.num_hours, self.num_stations)

        self.inputHorizon = 12 # number of time steps as input
        self.inOutVecDim = self.num_stations  # number of stations: Need to change depending on data!!!
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        #file_dataset = 'C:/Users/Jaime/Desktop/Minerva/workspace/nrel/Deep-Forecast/MS_winds.dat'
        self.file_dataset = file #trying
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
        self.activation = activation[1] ##################################################### MODIFICATION FROM 2 TO 1 RELU TO TANH
        realRun = 1 #Change from 0 to 1 if you want to use a 0.5% of the data to train
        #          model number :           1   2   3   4   5   6
        self.epochs, self.dataUsed = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.1]# percentage of data used for training(saving time for debuging)
        self.trainDataRate = 0.8
        self.num_hours = int(df.shape[0]*self.dataUsed)
        #print('Testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',self.epochs, self.trainDataRate) #I have changed from 1 to 0.715 since 6000 out of 
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
        trainRow = int(self.num_hours * self.trainDataRate) #Here we have the errors, datasets ad-hoc !!! Lets change it, instead of 6000 lets use percentages
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        # self.xTest = result[6000:6999, :]
        # self.yTest = self.winds[6000 + self.inputHorizon:6999 + self.inputHorizon]
        # testRow = int(self.num_hours * self.trainDataRate*2)
        self.xTest = X_train
        self.yTest = y_train
        print('\nNUM HOURS: ', self.num_hours)

        # self.xTest = result[trainRow:testRow, :]
        # self.yTest = self.winds[trainRow + self.inputHorizon:testRow+ self.inputHorizon]
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
        #print('AQUI 222 !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!', xTrain[:3][0], yTrain[:3][0])
        return [xTrain, yTrain]

  
    def buildModelLSTM_1(self):
        model = Sequential()
        in_nodes = out_nodes = self.inOutVecDim
        layers = [in_nodes, self.num_stations*2, self.num_stations, 32, out_nodes] #changing 57 to num_stations

        # model.add(LSTM(input_dim=layers[0],output_dim=layers[1],
        #     return_sequences=False))
        model.add(LSTM(units=layers[1], input_dim=layers[0],
            return_sequences=False))

        # model.add(Dense(
        #     output_dim=layers[4]))
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
        # lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, nb_epoch=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
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

    # def drawGraphStation(self, station, visualise = 1, ax = None ):
    #     '''draw graph of predicted vs real values'''

    #     yTest = self.yTest[:, station]
    #     denormalYTest = self.denormalize(yTest)

    #     denormalPredicted = self.denormalize(self.predicted[:, station])
    #     mae, rmse, nrmse_maxMin, nrmse_mean  = self.errorMeasures(denormalYTest, denormalPredicted)
    #     print('station %s : MAE = %7.7s   RMSE = %7.7s    nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, rmse, nrmse_maxMin, nrmse_mean ))

    #     if visualise:
    #         if ax is None :
    #             fig = plt.figure()
    #             ax = fig.add_subplot(111)

    #         ax.plot(denormalYTest, label='Real')
    #         ax.plot(denormalPredicted, label='Predicted', color='red')
    #         ax.set_xticklabels([0, 100, 200, 300], rotation=40)

    #     return mae, rmse, nrmse_maxMin, nrmse_mean

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
            rows, cols = 5, 10 # rows*cols < number of stations !!!
        elif self.num_stations >= 100:
            rows, cols = 10, 10 # rows*cols < number of stations !!!
        maeRmse = np.zeros((rows*cols,4))

        fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
        staInd = 0
        for ax in np.ravel(ax_array): 
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        plt.xticks([0, 100, 200, 300])#, rotation=45)
        errMean = maeRmse.mean(axis=0)
        print(errMean)
        #filename = 'pgf/finalEpoch'
        #plt.savefig('{}.pgf'.format(filename)) #ERROR
        #plt.savefig('{}.pdf'.format(filename))
        #plt.savefig('output/'+file.replace('.mat', '.png'))
        

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
        df = pd.read_csv(file, header=None) #Trying to parametrize
        self.num_stations = len(df.columns) #Trying to parametrize
        #print('Testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',self.num_hours, self.num_stations)

        self.inputHorizon = 12 # number of time steps as input
        self.inOutVecDim = self.num_stations  # number of stations: Need to change depending on data!!!
        self.lstmModels = [ None for _ in range(6)]
        self.xTest, self.yTest = None, None
        #file_dataset = 'C:/Users/Jaime/Desktop/Minerva/workspace/nrel/Deep-Forecast/MS_winds.dat'
        self.file_dataset = file #trying
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
        self.batchSize = 3 ####################################################################### MODIFICATION LSTM 3, from batch size of 3 to 64
        activation = ['sigmoid',   "tanh",   "relu", 'linear']
        self.activation = activation[2]
        realRun = 1 #Change from 0 to 1 if you want to use a 0.5% of the data to train
        #          model number :           1   2   3   4   5   6
        self.epochs, self.dataUsed = [[15, 17, 15, 17, 15, 15], 1] if realRun else [[ 1, 1, 1, 1, 1, 1] , 0.1]# percentage of data used for training(saving time for debuging)
        self.trainDataRate = 0.8
        self.num_hours = int(df.shape[0]*self.dataUsed)
        #print('Testing !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!',self.epochs, self.trainDataRate) #I have changed from 1 to 0.715 since 6000 out of 
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
        trainRow = int(self.num_hours * self.trainDataRate) #Here we have the errors, datasets ad-hoc !!! Lets change it, instead of 6000 lets use percentages
        X_train = result[:trainRow, :]
        y_train = self.winds[self.inputHorizon:trainRow + self.inputHorizon]
        # self.xTest = result[6000:6999, :]
        # self.yTest = self.winds[6000 + self.inputHorizon:6999 + self.inputHorizon]
        # testRow = int(self.num_hours * self.trainDataRate*2)
        self.xTest = X_train
        self.yTest = y_train
        print('\nNUM HOURS: ', self.num_hours)

        # self.xTest = result[trainRow:testRow, :]
        # self.yTest = self.winds[trainRow + self.inputHorizon:testRow+ self.inputHorizon]
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

        # model.add(LSTM(input_dim=layers[0],output_dim=layers[1],
        #     return_sequences=False))
        model.add(LSTM(units=layers[1], input_dim=layers[0],
            return_sequences=False, dropout=0.2))

        # model.add(Dense(
        #     output_dim=layers[4]))
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
        # train first LSTM with inputHorizon number of real input values

        lstmModel = self.buildModelLSTM(lstmModelNum)
        # lstmModel.fit(xTrain, yTrain, batch_size=self.batchSize, nb_epoch=self.epochs[lstmModelNum-1], validation_split=self.validation_split)
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

    # def drawGraphStation(self, station, visualise = 1, ax = None ):
    #     '''draw graph of predicted vs real values'''

    #     yTest = self.yTest[:, station]
    #     denormalYTest = self.denormalize(yTest)

    #     denormalPredicted = self.denormalize(self.predicted[:, station])
    #     mae, rmse, nrmse_maxMin, nrmse_mean  = self.errorMeasures(denormalYTest, denormalPredicted)
    #     print('station %s : MAE = %7.7s   RMSE = %7.7s    nrmse_maxMin = %7.7s   nrmse_mean = %7.7s'%(station+1, mae, rmse, nrmse_maxMin, nrmse_mean ))

    #     if visualise:
    #         if ax is None :
    #             fig = plt.figure()
    #             ax = fig.add_subplot(111)

    #         ax.plot(denormalYTest, label='Real')
    #         ax.plot(denormalPredicted, label='Predicted', color='red')
    #         ax.set_xticklabels([0, 100, 200, 300], rotation=40)

    #     return mae, rmse, nrmse_maxMin, nrmse_mean

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
            rows, cols = 5, 10 # rows*cols < number of stations !!!
        elif self.num_stations >= 100:
            rows, cols = 10, 10 # rows*cols < number of stations !!!
        maeRmse = np.zeros((rows*cols,4))

        fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
        staInd = 0
        for ax in np.ravel(ax_array): 
            maeRmse[staInd] = self.drawGraphStation(staInd, visualise=1, ax=ax)
            staInd += 1
        plt.xticks([0, 100, 200, 300])#, rotation=45)
        errMean = maeRmse.mean(axis=0)
        print(errMean)
        #filename = 'pgf/finalEpoch'
        #plt.savefig('{}.pgf'.format(filename)) #ERROR
        #plt.savefig('{}.pdf'.format(filename))
        #plt.savefig('output/'+file.replace('.mat', '.png'))
        

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
    #all_models.append(DeepForecast1), all_models.append(DeepForecast2), all_models.append(DeepForecast2)
    return all_models, [results1, results2, results3]

# create stacked model input dataset as outputs from the ensemble
# def stacked_dataset(members):
#     models_predictions_train = np.empty((len(members[0].yTest), members[0].num_stations, len(members)), float) #np.zeros((self.xTest.shape, len(members)))
#     models_predictions_val = np.empty((len(members[0].yVal), members[0].num_stations, len(members)), float) #np.zeros((self.xTest.shape, len(members)))
#     preds_train = []
#     preds_val = []
#     for model in members:
#         num_model = 0
#         model_prediction_train = np.empty((len(members[0].yTest), model.num_stations), float)
#         model_prediction_val = np.empty((len(members[0].yVal), model.num_stations), float)

#         #model_prediction = []
#         for station in range(model.num_stations):
#             station_prediction_train = model.denormalize(model.predicted[:, station])
#             station_prediction_val = model.denormalize(model.predictedVal[:, station])
#             print('testing station prediction train  and model prediction shape: ', station_prediction_train.shape, model_prediction_train[:, station].shape)
#             model_prediction_train[:, station] = station_prediction_train
#             model_prediction_val[:, station] = station_prediction_val

#         # models_predictions_train[:, :, num_model] = model_prediction_train
#         # models_predictions_val[:, :, num_model] = model_prediction_val #THIS SEEMS TO NOR WORK
#         print('TESTING  per model prediction TRAIN, VAL: ', model_prediction_train.shape, model_prediction_val.shape)
#         preds_train.append(model_prediction_train.flatten())
#         preds_val.append(model_prediction_val.flatten())
#     # (num_hours, num_stations, model) => (model1, model2, model3)

#     preds_train = np.asarray(preds_train).T
#     preds_val = np.asarray(preds_val).T
#     print('TESTING TRAIN, VAL: ', preds_train.shape, preds_val.shape)

#     # TO FOLLOW TOMORROW: SHAPESModels shape before reshape train:  (377, 57, 3)Models shape val:  (72, 57) THE VAL PREDICTIONS ARE WRONG, MAYBE SOME MODEL IS NOT RETURNING THEM

#     return preds_train, preds_val
def stacked_dataset(members):
    models_predictions_train = np.empty((len(members[0].yTest), members[0].num_stations, len(members)), float) #np.zeros((self.xTest.shape, len(members)))
    models_predictions_val = np.empty((len(members[0].yVal), members[0].num_stations, len(members)), float) #np.zeros((self.xTest.shape, len(members)))

    num_model = 0
    for model in members:
        model_prediction_train = np.empty((len(members[0].yTest), model.num_stations), float)
        model_prediction_val = np.empty((len(members[0].yVal), model.num_stations), float)

        #model_prediction = []
        for station in range(model.num_stations):
            station_prediction_train = model.denormalize(model.predicted[:, station])
            station_prediction_val = model.denormalize(model.predictedVal[:, station])
            model_prediction_train[:, station] = station_prediction_train
            model_prediction_val[:, station] = station_prediction_val

        models_predictions_train[:, :, num_model] = model_prediction_train
        models_predictions_val[:, :, num_model] = model_prediction_val #THIS SEEMS TO NOR WORK
        num_model+=1
        # print('TESTING  per model prediction TRAIN, VAL: ', model_prediction_train.shape, model_prediction_val.shape)
        # preds_train.append(model_prediction_train.flatten())
        # preds_val.append(model_prediction_val.flatten())
    # (num_hours, num_stations, model) => (model1, model2, model3)

    # preds_train = np.asarray(preds_train).T
    # preds_val = np.asarray(preds_val).T
    # print('TESTING TRAIN, VAL: ', preds_train.shape, preds_val.shape)

    # TO FOLLOW TOMORROW: SHAPESModels shape before reshape train:  (377, 57, 3)Models shape val:  (72, 57) THE VAL PREDICTIONS ARE WRONG, MAYBE SOME MODEL IS NOT RETURNING THEM
    models_predictions_train_norm = (models_predictions_train - members[0].windMin) / members[0].windMax
    models_predictions_val_norm = (models_predictions_val - members[0].windMin) / members[0].windMax
    return models_predictions_train_norm, models_predictions_val_norm
    
# fit a model based on the outputs from the ensemble members
def fit_stacked_model(members):
    from sklearn.linear_model import LinearRegression
    # create dataset using ensemble
    models_predictions_train_norm, models_predictions_val_norm = stacked_dataset(members)
    yTest = members[0].yTest
    #yTest = yTest.flatten()
    #print(type(yTest))
    # fit standalone model
    models = []
    print('\n=========================================================================================================')
    print('\nTraining Linear Regression models (one per station)...\n')
    print('=========================================================================================================\n')
    for station in range(members[0].num_stations):
        model = LinearRegression()
        model.fit(models_predictions_train_norm[:,station,:], yTest[:,station])
        models.append(model)
    return models

# def multi_nn(members, models_predictions):
#     model = Sequential()
#     num_stations = members[0].num_stations
#     layers = [num_stations, 30, num_stations * 2, num_stations, num_stations]
#     model.add(LSTM(units=layers[1], input_dim=models_predictions.shape, 
#         return_sequences=False, dropout=0.2))
#     model.add(Dense(units=layers[4]))
#     model.add(Activation('tanh'))

#     optimizer = optimizers.RMSprop(lr=0.001)
#     model.compile(loss="mae", optimizer=optimizer)
#     return model

# def multi_nn_pred(members, models_predictions):
#     nn = multi_nn(members, models_predictions)
#     models_predictions = stacked_dataset(members)
#     yTest = members[0].yTest
#     nn.fit(models_predictions, yTest, epochs=1, shuffle=False, verbose=0)
#     yhat = nn.predict(models_predictions, verbose=0)
#     return yhat


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
    np.save('output/stacked_'+file,denormalPredicted)
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
    #yPred = yPred.reshape(members[0].yVal.shape[0], num_stations)
    #yTest = yTest.reshape(members[0].yVal.shape[0], num_stations)
    if num_stations <= 57:
        rows, cols = 5, 10 # rows*cols < number of stations !!!
    elif num_stations >= 100:
        rows, cols = 10, 10 # rows*cols < number of stations !!!
    maeRmse = np.zeros((rows*cols,4))

    fig, ax_array = plt.subplots(rows, cols, sharex=True, sharey=True )
    staInd = 0
    for ax in np.ravel(ax_array): 
        maeRmse[staInd] = drawGraphStation_stacked(members, yPred, yTest, staInd, visualise=1, ax=ax)
        staInd += 1
    plt.xticks([0, 100, 200, 300])#, rotation=45)
    errMean = maeRmse.mean(axis=0)
    print(errMean)
    #filename = 'pgf/finalEpoch'
    #plt.savefig('{}.pgf'.format(filename)) #ERROR
    #plt.savefig('{}.pdf'.format(filename))
    #plt.savefig('output/'+file.replace('.mat', '.png'))

    return errMean

original = False
if original:
    file = 'MS_winds.mat'
    # DeepForecaste = multiLSTM(file)
    # DeepForecaste.run()
    members, results = load_all_models('data/'+file)
    #models_predictions_train,  models_predictions_val= stacked_dataset(members)
    stacked_models = fit_stacked_model(members)
    yPred = stacked_prediction(members, stacked_models)
    print('\n STACKED MODEL RESULTS \n')
    errMean = drawGraphAllStations_stacked(members, yPred, members[0].yVal, members[0].num_stations)
    print('\n')
    # Member 0 results: [ 2.94562247  3.57925937 35.09488742 96.72237862]
    # Member 1 results: [ 2.67331025  3.40230863 33.32712637 91.52410241]
    # Stacked results (EUREKA!!!): [ 1.21642792  1.57042633 15.41357204 42.64994741]
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

            df = pd.read_csv('output/results_def.csv')
            id = file.replace('.mat', '')
            
            # df.loc[df.dataset == id, 'mae'] = results[0][0]
            # df.loc[df.dataset == id, 'rmse'] = results[0][1]
            # df.loc[df.dataset == id, 'nrmse_mean'] = results[0][2]
            # df.loc[df.dataset == id, 'nrmse_maxMin'] = results[0][3]

            # df.loc[df.dataset == id, 'mae_2'] = results[1][0]
            # df.loc[df.dataset == id, 'rmse_2'] = results[1][1]
            # df.loc[df.dataset == id, 'nrmse_mean_2'] = results[1][2]
            # df.loc[df.dataset == id, 'nrmse_maxMin_2'] = results[1][3]

            # df.loc[df.dataset == id, 'mae_3'] = results[2][0]
            # df.loc[df.dataset == id, 'rmse_3'] = results[2][1]
            # df.loc[df.dataset == id, 'nrmse_mean_3'] = results[2][2]
            # df.loc[df.dataset == id, 'nrmse_maxMin_3'] = results[2][3]

            # df.loc[df.dataset == id, 'mae_stacked'] = errMean[0]
            # df.loc[df.dataset == id, 'rmse_stacked'] = errMean[1]
            # df.loc[df.dataset == id, 'nrmse_mean_stacked'] = errMean[2]
            # df.loc[df.dataset == id, 'nrmse_maxMin_stacked'] = errMean[3]
            row = {'id':id ,'mae': results[0][0], 'rmse': results[0][1], 'nrmse_mean': results[0][2], 'nrmse_maxMin':results[0][3], 'mae_2': results[1][0], 'rmse_2': results[1][1], 'nrmse_mean_2': results[1][2], 'nrmse_maxMin_2':results[1][3], 'mae_3': results[2][0], 'rmse_3': results[2][1], 'nrmse_mean_3': results[2][2], 'nrmse_maxMin_3':results[2][3], 'mae_stacked': errMean[0], 'rmse_stacked': errMean[1], 'nrmse_mean_stacked': errMean[2], 'nrmse_maxMin_stacked': errMean[3]}
            df = df.append(row, ignore_index = True)
            df.to_csv('output/results_def.csv', index = False)
            #ids.append(file.replace('.mat', '')), mae.append(errMean[0]), rmse.append(errMean[1]), nrmse_maxMin.append(errMean[2]), nrmse_mean.append(errMean[3])
            contador+=1
# means copy paste in excel results.csv: 4.783423201	4.231984641	4.73229977	1.387896327	5.697113444	5.22084601	5.659446881	1.840254581	36.20262953	33.07274125	35.97523014	11.73074147	95.38847758	88.38208312	94.98086852	33.53231674

    

    # df = pd.read_csv('output/results_def.csv')
    # for id in ids:
    #     df.loc[df.id == id, 'mae_stacked'] = mae
    #     df['mae_stacked'], df['rmse_stacked'], df['nrmse_maxMin_stacked'], df['nrmse_mean_stacked']= mae, rmse, nrmse_maxMin, nrmse_maxMin
    # df.to_csv('output/results.csv')

# l=inputHorizon=12 (se usan las 12 horas previas para predecir) h=6 (se predice en intervalos de 6 horas) Forecasting performance on 16 stations
# Probably each model is for 1 hour ahead

# PARECE QUE yVAL esta normalizado entre 0 y 1, mientras que los yPred de las redes neuronales no. CHECKEAR!!!!