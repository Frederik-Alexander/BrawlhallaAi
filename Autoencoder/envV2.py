from PIL import Image
import pyautogui as gui
import numpy as np
import time,os,colorsys
import autoit
import mss


class BrawlahallaEnv:
    # Initialize Brawlhalla Environment
    def __init__(self):
        self.dir = os.path.dirname(__file__)
        l = 800
        self.size = (l,l) # edges of window
        width,height = self.size
        self.setWindow(0,0,(width+8+8,height+31+8)) 

        self.window = (8,31,width+8,height+31)
        self.sct = mss.mss()

        self.keys = [['w','a','s','d','y',''],[0,0,0,0,0,0]] #"y" = shift
        self.hpPixels = ( (int(width*(365-8)/400), int(height*(53-31)/400)) , (int(width*(389-8)/400), int(height*(53-31)/400)))
        self.liveLostPixels = ( (int(width*(1094-8)/1200), int(height*(72-31)/1200)) , (int(width*(1168-8)/1200), int(height*(68-31)/1200)))

        self.episode_running = False
        self.frame = 0
        self.h,self.s,self.v = [0,0],[0,0],[0,0]

        self.updateScreenshot()
        self.damageTaken(0)
        self.damageTaken(1)
        print("Running")

    # sets game size and position
    def setWindow(self, x, y, size):
        w, h = size
        print("Set window to w: {}, h: {}".format(w, h))
        autoit.win_move("Brawlhalla", x, y, w, h)

    # proceed to next game frame
    def nextFrame(self): # each keypress(up or down) takes 0.11s
        for i in range(10):
            autoit.control_send("", "", "{F6}")
            self.frame += 1

    # update screenshot variable
    def updateScreenshot(self): #returns greyscale screenshot, takes 0.4 secs
        self.screenshot = Image.fromarray(np.array(self.sct.grab(self.window)))

    # returns if a Player is full Health
    def fullHp(self,position):
        r,g,b = self.screenshot.getpixel(self.hpPixels[position])[0:3]
        if r == g == b == 255:
            return True
        return False

    # returns if a player has taken damage since last time the function was called
    def damageTaken(self,position):
        r,g,b = self.screenshot.getpixel(self.hpPixels[position])[0:3]
        h,s,v = colorsys.rgb_to_hsv(r,g,b) # grad zahl des Farbwertes
        if self.h[position]!=h or self.s[position]!=s or self.v[position]!=v:
            self.h[position] = h
            self.s[position] = s
            self.v[position] = v
            return True
        return False
 
    # executes action array as Keypresses
    def pressKeys(self,keyArray): # input array of "keys" length, True or False
        for i in range(len(self.keys[0])):
            if keyArray[i] == self.keys[1][i]:
                continue

            if keyArray[i]:
                print("Sending!: " +self.keys[0][i] )  
                autoit.control_send("", "", "{"+self.keys[0][i]+" down}" )
            else:
                print("Stopping: " +self.keys[0][i] )
                autoit.control_send("", "", "{"+self.keys[0][i]+" up}" )

            
            self.keys[1][i] = keyArray[i] 

    # returns if the AI has died  
    def is_episode_finished(self,position):
        r,g,b = self.screenshot.getpixel(self.liveLostPixels[position][0:2])[0:3]
        h,s,v = colorsys.rgb_to_hsv(r,g,b) # grad zahl des Farbwertes
        hue = 360*h
        # print (hue)
        if 360 >= hue > 230:
            self.episode_running = False
            return True
        return False

    # start new Episode
    def new_Episode(self):
        print("New Episode")
        time.sleep(1)
        self.episode_running = True
