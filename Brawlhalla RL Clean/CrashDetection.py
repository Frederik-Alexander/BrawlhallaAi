# -*- coding: utf-8 -*-

import os
import telepot
import subprocess

class ProcessState:
    def __init__(self,processName):
        self.processName = processName

    def getTasks(self,name):
        for task in os.popen('tasklist /fi "IMAGENAME eq {}"'.format(self.processName)):
            #print(task)
            if name in task:
                #print(task)
                return task
        return None
    
    def isAlive(self):

        r = self.getTasks(self.processName)

        if not r:
            return False
        elif 'Not Responding' in r:
            return False            
        else:
            return True

class telegram:
    def __init__(self,API_KEY):
        self.bot = telepot.Bot(API_KEY)

    def handle(self, msg):
        print (msg)

    def start_listener(self):
        from telepot.loop import MessageLoop
        MessageLoop(self.bot,self.handle).run_as_thread()

    def send(self, chatId, msg):
        self.bot.sendMessage(chatId, msg)

gameState = ProcessState("Brawlhalla.exe")

print(gameState.isAlive())