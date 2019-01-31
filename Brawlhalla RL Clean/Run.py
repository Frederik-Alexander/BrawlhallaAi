#runs realtime

#1. get model
#2. setup env
#3. 

import random
import time
import numpy as np

from envV2 import BrawlahallaEnv #is in helperfunc
import HelperFunctions #preproccess...
from DQNetwork import DQNetwork

print("Setting Up Enviroment")
game, possible_actions = HelperFunctions.create_environment()


#Hyperparameters:


stack_frame_size = 10 #Stack Frames to overcome temproal limitaion of model 
unencoded_state_size = (300,300) # size of screenshot !!FIRST HEIGHT!!
state_size = (150,150)


#NN Hyperparameters:
action_size = 8
input_shape = (state_size[0],state_size[1],stack_frame_size)

Load_Model = True

print("Setting Up Stacked Frames Deque")
stacked_frames = HelperFunctions.stack_frames(stack_frame_size,state_size,unencoded_state_size)


print("Setting Up DQNN")
DQNetwork = DQNetwork(input_shape,action_size ,Load_Model)


###############################
while 1:
    #game.nextFrame()
    game.updateScreenshot()
    frame = game.screenshot
    t1 = time.time()

    # Stack the frames
    stacked_frames.add(frame)
    state = stacked_frames.get()#should return np array


    #Predict Action
    predicted_action = DQNetwork.predict( [ np.array( [state] )] )
    #print("predicted action vector: " + str(predicted_action))

    avg = np.mean(predicted_action)
    print(avg)


    #Execute Action
    action_index = np.argmax(predicted_action)
    predicted_action = np.zeros((action_size,),dtype=int)
    predicted_action[action_index] = 1
    print(predicted_action)
    #Execute Action
    game.pressKeys(predicted_action)
    #rint(time.time()-t1)
    
    