#Memory function:
from collections import deque
import numpy as np
import random
import os.path
import pickle

memory_path = "C:/Users/Ai/Desktop/Brawlhalla RL/MEMORY.obj"

class Memory_array():

    def __init__(self, memory_size, Load_Memory):
        if Load_Memory == True  and os.path.isfile(memory_path) == True:
            
            filehandler = open(memory_path, 'rb')
            self.Memory = pickle.load(filehandler)

            print("Loaded Memory")
            print(type(self.Memory))
            
        else:
            self.Memory = deque(maxlen = memory_size) 
            print("Created Memory")
            print(type(self.Memory))
            

    def add(self,value):
        self.Memory.append(value)
    

    def sample(self, batch_size):
        buffer_size = len(self.Memory)

        index = np.random.choice(np.arange(buffer_size), size = batch_size, replace = False)
        return [self.Memory[i] for i in index]


    def saveMemory(self):
        filehandler = open(memory_path, 'wb')
        pickle.dump(self.Memory,filehandler)



    def preFill_Memory(self, already_created_Memory, game, possible_actions, pretrain_length, stacked_frames ):
                
        # Render the environment
        game.new_Episode()

        for i in range(pretrain_length+1):#because i =0 doesnt add a memory
            print(i)
            game.updateScreenshot()

            # If it's the first step
            if i == 0:
                frame = game.screenshot
                stacked_frames.new_episode(frame)
                state = stacked_frames.get()#should return np array #gets first state



            # Random action 
            action = random.choice(possible_actions)
            print(action)


            game.pressKeys(action)#do action
            game.nextFrame() 
            game.updateScreenshot()






            if game.damageTaken(1):
                reward = -30 #temp
                print("TOOK DMG -30 REWARD!") #works
            else:
                reward = 0.10


            # Look if the episode is finished
            done = game.is_episode_finished(1)
            
            # If we're dead
            if done:
                # We finished the episode
                next_state = np.zeros(state.shape)
                
                # Add experience to memory
                already_created_Memory.add((state, action, reward, next_state, done))
                
                # Start a new episode
                game.new_Episode()
                
                # First we need a state
                game.nextFrame()
                game.updateScreenshot()
                frame = game.screenshot
                # Stack the frames
                stacked_frames.new_episode(frame)
                state = stacked_frames.get()
                
            else:
                # Get the next state
                game.updateScreenshot()
                frame = game.screenshot
                stacked_frames.add(frame)
                next_state = stacked_frames.get() # get state t+1
                
                # Add experience to memory
                already_created_Memory.add((state, action, reward, next_state, done))
                
                # Our state is now the next_state
                state = next_state 

        print("filled memory with " + str(pretrain_length) + " frames (state, action, reward, next_state, done) ")

        return already_created_Memory



