import pythoncom
import pyHook as ph
from PIL import Image
import numpy as np
import time,os
import autoit
import mss
import threading


class game_env():
    def __init__(self,windowSize=1000):
        width,height = (windowSize,windowSize)
        self.setWindow(0,0,(width+8+8,height+31+8)) 
        # Windows adds an 8px Frame around a window
        # The Brawlhalla game window is now the desired size.

        self.window = (8,31,width+8,height+31)
        self.sct = mss.mss()

        self.updateScreenshot()

    def setWindow(self,x,y,size):
        w,h = size
        print("Set window to w: {}, h: {}".format(w,h))
        try:
            autoit.win_move("Brawlhalla",x,y,w,h)
        except:
            print("Window could not be moved. Exiting")
            exit()

    def updateScreenshot(self):
        self.screenshot = Image.fromarray(np.array(self.sct.grab(self.window)))
        return self.screenshot


class key_handler():
    def __init__(self):
        hm = ph.HookManager()
        hm.KeyDown = self.down_Hook
        # hm.KeyUp = self.up_Hook
        hm.HookKeyboard()
        self.order = ['38','37','40','39','y','x']
        self.key_stats = {'38':0,'37':0,'40':0,'39':0,'y':0,'x':0}
        # up,left,down,right,dodge,light

    def keys_to_array(self):
        x = []
        for pos in self.order:
            x.append(self.key_stats[pos])
        return x

    def down_Hook(self,event):
        # runs in thread
        self.updateKeys(event.KeyID,1)
        return True

    def up_Hook(self,event):
        # runs in thread
        self.updateKeys(event.KeyID,0)
        return True

    def updateKeys(self,keyID,state):
        for key in self.key_stats:
            self.key_stats[key] = 0
        # alphabet begins at Id 65 and ends at 90
        if keyID < 65 or keyID > 90:
            # if not letter
            key = str(keyID)
        else:
            key = chr(keyID).lower()
            if key == 'q':
                print("Exit")
                exit()

        if key in self.key_stats:
            self.key_stats[key] = state
        print(self.key_stats)

        # print(self.keys_to_array())
    
    def getKeys(self):
        pythoncom.PumpWaitingMessages()
        return self.keys_to_array()


def savepicture(pic,path):
    pic = pic.convert('L')
    #print(np.asarray(pic).shape)
    #exit()
    state_size = (400,400)
    pic = pic.resize(state_size)
    pic.save(path)
    return None


path = os.path.dirname(__file__)
print("Current directory: "+path)
key_handler = key_handler()

# t = time.time()
# while time.time() - t < 3:
#     key_handler.getKeys()

# Module Initialization
game_env = game_env(windowSize=800)

time1 = str(time.strftime("%Y.%m.%d_%H-%M-%S"))
os.makedirs(path +  "/Data/" + time1) 
data_path = path +"/Data/" + time1 + "/"

pic_counter_id = 0

print("Loop running...")
while True:
    pic_counter_id+=1
    t1 = time.time()
    # dict of keys and current status
    keys_array = key_handler.getKeys()

    # screenshot of game window
    screen = game_env.updateScreenshot()

    pic_path = data_path + str(pic_counter_id) + " " + str(keys_array) + ".png"
    t = threading.Thread(target=savepicture,args=(screen,pic_path))
    t.start()

    while time.time()-t1 < 1/80:
        key_handler.getKeys()