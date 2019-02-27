#keras DQNN
import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras import optimizers
import numpy as np
import os 

model_weights_path = 'C:/Users/Ai/Desktop/Brawlhalla RL/model_weights.h5'

class DQNetwork():
    def __init__(self, input_shape, action_size, load_Model):


        """        self.model = Sequential()
        self.model.add(Dense(input_shape=input_shape,units=10, activation='relu'))
        self.model.add(Flatten())
        self.model.add(Dense(1, activation='relu'))
        
        self.model.add(Dense(action_size))"""
        

        self.model = Sequential()
        self.model.add(Conv2D(32, kernel_size=(8, 8), strides=(1, 1),
                        activation='relu',
                        input_shape=input_shape))

        #self.model.add(MaxPooling2D(pool_size=(2, 2)))

        self.model.add(Conv2D(64, (4, 4), activation='relu', strides=(2, 2)))
        #self.model.add(MaxPooling2D(pool_size=(2, 2)))
        

        self.model.add(Conv2D(128, (2, 2), activation='relu', strides=(2, 2)))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Flatten())

        self.model.add(Dense(512, activation='relu'))
        self.model.add(Dense(action_size))







        import tensorflow as tf
        run_opts = tf.RunOptions(report_tensor_allocations_upon_oom = True)

        self.model.compile(loss=keras.losses.mean_squared_error,
              optimizer=keras.optimizers.RMSprop(lr=0.000002),#lr=0.0002
              metrics=['accuracy'],options = run_opts)

        if load_Model == True  and os.path.isfile(model_weights_path) == True:
            print("Loading models")
            self.load()

    def predict(self,state):
        predicted_actions = self.model.predict(state)
        return predicted_actions

    
    def train(self,batch):
        pass
    def save(self):
        #os.remove(model_weights_path)
        self.model.save_weights(model_weights_path)
        pass
    def load(self):
        self.model.load_weights(model_weights_path)
        

        pass

#TESTING SECTION!
#we input a state with dimensions : 1200,560,4
#action size 
"""action_size = 4
input_shape = (570,1200,4)
#input_shape = (10,10,4)

DqNNtest = DQNetwork(input_shape,action_size)

test_array = np.zeros((570,1200,4), dtype = int)

print(max(max(  DqNNtest.predict( [ np.array( [test_array] )] )  )))
print(DqNNtest.predict( [ np.array( [test_array] )] ).shape  )
print(max([1,1,2,2]))"""