# -*- coding: utf-8 -*-

import os
import telepot
import subprocess


class ProcessState:
    # Initialize with process name
    # in this Case "Brawlhalla.exe"
    def __init__(self, processName):
        self.processName = processName

    # get system tasklist and return process with "name" in it
    def getTasks(self, name):
        for task in os.popen('tasklist /fi "IMAGENAME eq {}"'.format(name)):
            if name in task:
                return task
        return None

    # return process state as bool
    def isAlive(self):
        r = self.getTasks(self.processName)
        if not r:
            return False
        elif 'Not Responding' in r:
            return False            
        else:
            return True


class telegram:
    # setup telegram Bot with key
    def __init__(self, API_KEY):
        self.bot = telepot.Bot(API_KEY)

    def handle(self, msg):
        print(msg)

    # detect incoming messages
    def start_listener(self):
        from telepot.loop import MessageLoop
        MessageLoop(self.bot,self.handle).run_as_thread()

    # send message
    def send(self, chatId, msg):
        self.bot.sendMessage(chatId, msg)
