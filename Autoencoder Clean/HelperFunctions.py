#Helper Functions 

import numpy as np
from collections import deque
from envV2 import BrawlahallaEnv 

def preprocess_frame(frame, state_size):# convert frame to greyscale and normalize values

    img = frame.convert('L')
    

    # Normalize Pixel Values
    #normalized_frame = grey_image/255.0 #useless!!!
    #FIRST CROP
    width, height = img.size   # Get dimensions
    new_width = width-50
    new_height = height-50

    left = (width - new_width)/2
    top = (height - new_height)/2
    right = (width + new_width)/2
    bottom = (height + new_height)/2

    img = img.crop((left, top, right, bottom))

    #THEN RESIZE

    state_size=state_size
    img = img.resize(state_size)

    

    grey_image = np.array(img)

    #print(np.array(img).shape)
    #img.show()
    normalized_frame = grey_image
    
    #tempend

    preprocessed_frame = normalized_frame
    return preprocessed_frame



class stack_frames():
    def __init__(self, deque_size, state_size):
        self.state_size_local = state_size
        self.stack_site_local = deque_size

        self.stack_frames_Deque = deque([np.zeros((self.state_size_local), dtype=np.int) for i in range(self.stack_site_local)], maxlen=self.stack_site_local)
        
        

    def new_episode(self, frame):
        self.stack_frames_Deque = deque([np.zeros((self.state_size_local), dtype=np.int) for i in range(self.stack_site_local)], maxlen=self.stack_site_local)
        frame = preprocess_frame(frame,self.state_size_local )
        self.stack_frames_Deque.append(frame)
        self.stack_frames_Deque.append(frame)
        self.stack_frames_Deque.append(frame)
        self.stack_frames_Deque.append(frame)

    def add(self, frame):
        frame = preprocess_frame(frame, self.state_size_local)
        self.stack_frames_Deque.append(frame)


    def get(self):
        
        """        from PIL import Image
        frame = np.stack(self.stack_frames_Deque, axis=0)[1]
        print(frame.shape)
        img = Image.fromarray(frame,'L')
        
        img.show('LA')
        
        #TEMP
        exit()"""

        return np.stack(self.stack_frames_Deque, axis=2)

def create_environment():#dswswdasdswaws
    game = BrawlahallaEnv()


    # Here our possible actions
    w = [1, 0, 0, 0, 0, 0]
    a = [0, 1, 0, 0, 0, 0]
    s = [0, 0, 1, 0, 0, 0]
    d = [0, 0, 0, 1, 0, 0]
    shift = [0, 0, 0, 0, 1, 0]
    null = [0, 0, 0, 0, 0, 1]
    possible_actions = [w, a, s, d, shift, null]

    return game, possible_actions

"""test_deque = stack_frames(4,(1200,570))

print(np.array(test_deque.stack_frames_Deque).shape)
"""
