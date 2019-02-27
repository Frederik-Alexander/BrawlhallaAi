import os
import json
import sys
import numpy as np
import random
import cv2

import keras
import random

###networks###
import importlib


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))


class Train_autoencoder():
    def __init__(self, input_shape, learning_rate, autoencoder_num=1):
        self.input_shape = input_shape
        self.autoencoder_num = autoencoder_num
        self.learning_rate = learning_rate
        
        module_name = "AUTOENCODER_"+str(autoencoder_num)
        #module_name ="AUTOENCODER"

        _temp  = importlib.__import__(module_name)
        
        #Autoencoder =_temp.Autoencoder
        #Autoencoder = Autoencoder.Autoencoder
        #import Autoencoder_module.Autoencoder

        
        bottleneck_dim = 64

        self.AutoencoderCLASS = _temp.Autoencoder(input_shape,10,learning_rate,True)
        self.Autoencoder = self.AutoencoderCLASS.autoencoder


        ###process_netowrk:###

        #NeuralNetwork = NeuralNetwork((300,300,stack_size),6,True)
        #self.NeuralNetwork = NeuralNetwork


    def fit(self,training_data, Nn_batch_size=20):
        X_train, Y_train, stack_size = training_data
        
        ###REFORMATING OF DATA###
    
        Y = []
        for i in Y_train:
            Y.append(random.choice(i))

        #Y = [np.array(Y)]
        #X = [np.array(X_train)]


        X = X_train

        #test train split
        x_train = X[:int((len(X)*0.8))]
        x_test = X[-(int(len(X)*0.2)):]

        #print(x_train[0].shape)
        #exit()

        x_train = np.array(x_train)
        x_test =  np.array(x_test)

        print(x_train.shape)
        print(x_test.shape)
        x_train = x_train.astype('float32') / 255.
        x_test = x_test.astype('float32') / 255.
        #x_train = np.reshape(x_train, (len(x_train), 300, 300 ))  # adapt this if using `channels_first` image data format
        #x_test = np.reshape(x_test, (len(x_test), 300, 300))  # adapt this if using `channels_first` image data format
        original_dim = self.input_shape[0]*self.input_shape[1] #temp for feed forward VAE
        x_train = np.reshape(x_train, [-1, self.input_shape[0],self.input_shape[1],1])
        x_test = np.reshape(x_test,[-1, self.input_shape[0],self.input_shape[1],1])

        history = LossHistory()

        history = self.AutoencoderCLASS.train(x_train, x_test,history, epochs=5, batch_size=Nn_batch_size)#epochs=100000
        self.AutoencoderCLASS.save()

        path_grid_results = "C:/Users/Ai/Desktop/Brawlhalla Supervised/GRIDSEARCHRESULTS/Net_" + str(self.autoencoder_num) + " shape_"+ str(self.input_shape[1]) + " lr_" + str(self.learning_rate)
        self.AutoencoderCLASS.IMG_RESULT(path_grid_results, x_test,True, n = 3 ,figsize=(50,50) ,size = (self.input_shape[0],self.input_shape[1]) )
        return history
        #test_array = np.zeros((300,300,1), dtype = int)
        #NeuralNetwork.train([ np.array( [test_array] )]   , [ np.array([[0., 1., 2., 1, 3., 0. ]])  ]     )
        #DqNNtest.train([ np.array( [test_array,test_array]  )]   , [ np.array([[0., 1., 2., 1, 3., 0. ] ,[0., 1., 2., 1, 3., 0. ]])  ]     )"""


        