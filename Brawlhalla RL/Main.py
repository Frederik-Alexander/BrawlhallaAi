#Main SCRIPT 

#Import:
import random,threading
import time,datetime
import numpy as np
import sys
import os 

#Our Import:
from envV2 import BrawlahallaEnv #is in helperfunc
from MemoryArray import Memory_array
from Reward_Calculator import RewardCalc
import HelperFunctions #preproccess...
from CrashDetection import *

#Our DQNN: (keras)
from DQNetwork import DQNetwork


print("Setting Up Enviroment")
game, possible_actions = HelperFunctions.create_environment()


#Hyperparameters:
training_episodes = 1000

memory_size = 1000 #Size of memory Deque
stack_frame_size = 10 #Stack Frames to overcome temproal limitaion of model 
unencoded_state_size = (300,300) # size of screenshot !!FIRST HEIGHT!!
state_size = (150,150) #New addidion 
max_episode_len = 100000 #500 temp 50
batch_size = 100 #100 temp 1 

train_every_step_value = 10 #set 1 if to train at every step

#Exploration parameters for epsilon greedy strategy
explore_start = 0.5            # exploration probability at start
explore_stop = 0.01            # minimum exploration probability 
decay_rate = 0.002  #decay_rate = 0.0001         # exponential decay rate for exploration prob
decay_step = 0


#NN Hyperparameters:
action_size = 8
input_shape = (state_size[0],state_size[1],stack_frame_size)#old
input_shape = (150,150,stack_frame_size)#new since encoder
# Q learning hyperparameters
gamma = 0.98              # Discounting 

Load_Model = True # if true loads h5 model and resumes training 
Load_Memory = True # Imports Memory 

#for telegram
API_KEY = "725588660:AAFSIo8fTzP2UD4_tnqFWZ3Zec39uJKdunM"
chatId_Frederik = 409574701
messager = telegram(API_KEY)


gameState = ProcessState("Brawlhalla.exe")



###############################
print("Running Main Script...")
###############################





print("Setting Up Memory of size: "+ str(memory_size))
Memory = Memory_array(memory_size, Load_Memory) #load memory only works if there is a memory to be loaded



print("Setting Up Stacked Frames Deque")
stacked_frames = HelperFunctions.stack_frames(stack_frame_size,state_size, unencoded_state_size)


print("Setting Up DQNN")
DQNetwork = DQNetwork(input_shape,action_size ,Load_Model)



print("Setting up Reward Obj")#dunno why class??!! mabye reason idk
RewardCalc = RewardCalc()


print("Prefilling Memory")
Memory = Memory.preFill_Memory( Memory, game, possible_actions, batch_size, stacked_frames)



#TESTING METRIC
frames_alive = []
reward_sum = []
relative_reward = []
w_sum = []
a_sum = []
s_sum = []
d_sum = []
w_sum_relevated = []
a_sum_relevated = []
s_sum_relevated = []
d_sum_relevated = []

Q_sum = []
Q_sum_relevated = []


def scriptEnd():
    print("Saving Data..")
    DQNetwork.save()
    np.savetxt("C:/Users/Ai/Desktop/Brawlhalla RL/ALIVE_TIME_EVOLUTION.txt",np.array(frames_alive,dtype=int),fmt="%d")
    

    with open(metrics_path + "REWARD_SUM_NO_DELETE.txt",'a') as f:
        np.savetxt(f,np.array(reward_sum,dtype=float),fmt="%d")

    with open(metrics_path + "ALIVE_TIME_NO_DELETE.txt",'a') as f:
        np.savetxt(f,np.array(frames_alive,dtype=float),fmt="%d")
            
    print("Done saving Data.")
    print("Episodes: "+str(counter))
    print("Program Ended at: "+str(datetime.datetime.now()))
    exit()

memory_counter = 0
counter=0
t1 = time.time()
print("Initialising Training")
try:
    for i in range(0,training_episodes):

    
        print("Episode NR."+str(counter))
        reward_sum_ticker = 0
        Q_sum_ticker = 0 # not ticker but summer -f

        w_sum_ticker = 0
        a_sum_ticker = 0
        s_sum_ticker = 0
        d_sum_ticker = 0

        decay_step = decay_step+1
        
        print("New Episode")
        print("explore_probability: " + str(explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)))
        time.sleep(4)

        for i1 in range(0,max_episode_len+1):
    
            t1 = time.time()
            
            #check if game has crashed


            if not gameState.isAlive():
                messager.send(chatId_Frederik, "Game not responding")
                scriptEnd()
            if i1 == 0: 
                
                game.updateScreenshot()
                frame = game.screenshot
                
                stacked_frames.new_episode(frame) #flushes episode  

            else:
                game.updateScreenshot()
                frame = game.screenshot
                
                # Stack the frames
                stacked_frames.add(frame)
                state = stacked_frames.get()#should return np array
    



                #Predict Action
                predicted_action = DQNetwork.predict( [ np.array( [state] )] )
                # print("predicted action vector: " + str(predicted_action))

                #Q-Metrics
                Q_sum_ticker = Q_sum_ticker + np.mean(predicted_action )

                #Proccesses action vector 
                action_index = np.argmax(predicted_action)
                predicted_action = np.zeros((action_size,),dtype=int)
                predicted_action[action_index] = 1




                #Epsilon Greedy
                ## EPSILON GREEDY STRATEGY
                # Choose action a from state s using epsilon greedy
                ## First we randomize a number
                exp_exp_tradeoff = np.random.rand()

                # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
                explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)
                
                if (explore_probability > exp_exp_tradeoff):
                    # Make a random action (exploration)
                    predicted_action = random.choice(possible_actions)
                    # print("EPSILON GREEDY STRATEGY")




                # print(predicted_action)
                #Execute Action
                game.pressKeys(predicted_action)
                game.nextFrame()

                #metrics
                if predicted_action[0] == 1:
                    w_sum_ticker = w_sum_ticker+1
                if predicted_action[1] == 1:
                    a_sum_ticker = a_sum_ticker+1
                if predicted_action[2] == 1:
                    s_sum_ticker = s_sum_ticker+1
                if predicted_action[3] == 1:
                    d_sum_ticker = d_sum_ticker+1





                #Get Next State
                game.updateScreenshot()
                frame = game.screenshot
                


                # Stack the frames
                stacked_frames.add(frame)
                next_state = stacked_frames.get()

               



                #Get Reward
                
                #Hier die wichtigsten werte des Reward Engeneerings:
                if game.damageTaken(1):
                    reward = -10 #temp
                    #print("TOOK DMG -30 REWARD!") #works
                else:
                    reward = 4

                if game.damageTaken(0):
                    reward += 10

                done_enemy = game.is_episode_finished(0)
                if done_enemy:
                    reward = 100

                #check if over
                done = game.is_episode_finished(1)

                if done:
                    #end episode
                    reward = -500
                 

                
                #Store experience
                Memory.add((state, predicted_action, reward, next_state, done))
                print("Reward: "+ str(reward))
                memory_counter = memory_counter+1#metrics
                reward_sum_ticker = reward_sum_ticker + reward#metrics
                if done:
                    #end episode
                    print("Episode ended, player died")
                    frames_alive.append(i1)
                    reward_sum.append(reward_sum_ticker)
                    w_sum.append(w_sum_ticker)
                    a_sum.append(a_sum_ticker)
                    s_sum.append(s_sum_ticker)
                    d_sum.append(d_sum_ticker)

                    Q_sum.append(Q_sum_ticker)
                    break






                if i1 % train_every_step_value == 0: # training every step is to computer intensive

                

                    ########################
                    #Train NN with mini batch
                    ########################

                    batch = Memory.sample(batch_size) #mb = mini batch
                    states_mb = np.array([each[0] for each in batch], ndmin=3)
                    actions_mb = np.array([each[1] for each in batch])
                    rewards_mb = np.array([each[2] for each in batch]) 
                    next_states_mb = np.array([each[3] for each in batch], ndmin=3)
                    dones_mb = np.array([each[4] for each in batch])

                    target_Qs_batch = []
                    Qs_next_state = []

                    for i in next_states_mb:
                        Qs_next_state.append(DQNetwork.predict( [ np.array( [i] )] ))


                    # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma*maxQ(s', a')
                    for i in range(0, len(batch)):
                        terminal = dones_mb[i]

                        # If we are in a terminal state, only equals reward
                        if terminal:
                            target_Qs_batch.append(rewards_mb[i])
                            
                        else:
                            target = rewards_mb[i] + gamma * np.max(Qs_next_state[i])
                            target_Qs_batch.append(target)



                    targets_mb = np.array([ [each,each,each,each,each,each,each,each] for each in target_Qs_batch]) # x4 because the target doesent change since it uses max(Q(a)) not ever a


                    t = threading.Thread(target=game.nextFrame())
                    t.start()
                    DQNetwork.model.fit([states_mb],targets_mb)

                    while t.isAlive():
                        pass
                # print(time.time()-t1)




        else: # if max frames has passed
            frames_alive.append(max_episode_len)
            reward_sum.append(reward_sum_ticker)
            w_sum.append(w_sum_ticker)
            a_sum.append(a_sum_ticker)
            s_sum.append(s_sum_ticker)
            d_sum.append(d_sum_ticker)
            Q_sum.append(Q_sum_ticker)
        






        ######################################################################
        #SAVING EACH EPISODE
        ######################################################################




        metrics_path = "C:/Users/Ai/Desktop/Brawlhalla RL/METRICS/"
        #relative reward = reward/frames
        relative_reward = [reward_sum[i]/frames_alive[i] for i in range(0,len(reward_sum))]

        Q_sum_relevated =  [Q_sum[i]/frames_alive[i] for i in range(0,len(Q_sum))]


        w_sum_relevated =  [w_sum[i]/frames_alive[i]*100 for i in range(0,len(w_sum))] #Percentage of w presses
        a_sum_relevated =  [a_sum[i]/frames_alive[i]*100 for i in range(0,len(a_sum))]
        s_sum_relevated =  [s_sum[i]/frames_alive[i]*100 for i in range(0,len(s_sum))]
        d_sum_relevated =  [d_sum[i]/frames_alive[i]*100 for i in range(0,len(d_sum))]
    
        np.savetxt(metrics_path + "w_sum_relevated.txt",np.array(w_sum_relevated,dtype=float),fmt="%d")
        np.savetxt(metrics_path + "a_sum_relevated.txt",np.array(a_sum_relevated,dtype=float),fmt="%d")
        np.savetxt(metrics_path + "s_sum_relevated.txt",np.array(s_sum_relevated,dtype=float),fmt="%d")
        np.savetxt(metrics_path + "d_sum_relevated.txt",np.array(d_sum_relevated,dtype=float),fmt="%d")
        
        np.savetxt(metrics_path + "Q_sum_relevated.txt",np.array(Q_sum_relevated,dtype=float),fmt="%d")

        np.savetxt(metrics_path + "ALIVE_TIME.txt",np.array(frames_alive,dtype=float),fmt="%d")
        np.savetxt(metrics_path + "REWARD_SUM.txt",np.array(reward_sum,dtype=float),fmt="%d")
        np.savetxt(metrics_path + "RELATIVE_REWARD_SUM.txt",np.array(relative_reward,dtype=float),fmt="%d")
        


        

        print("saving.")
        print("Total Memory Added: " + str(memory_counter))
        DQNetwork.save() #temp comment

        #Memory.saveMemory()
        counter+=1


except Exception as e:
    print(e)
    exc_type, exc_obj, exc_tb = sys.exc_info() # for meta data
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)
except:
    scriptEnd()



print(time.time()-t1)
 