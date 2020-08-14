#!/usr/bin/env python3

import datetime as dt
from threading import Thread
import cv2

class FPS:

    def __init__(self):
        self.mystart = None
        self.myend = None
        self.numFrames = 0

    def start(self):
        self.mystart = dt.datetime.now()
        return self

    def stop(self):
        self.myend = dt.datetime.now()
        return self

    def update(self):
        self.numFrames += 1
        return self

    def elapsed(self):
        return (self.myend - self.mystart).total_seconds()

    def fps(self):
        return self.numFrames/self.elapsed()

class myVideoCapture:
    def __init__(self, src=0):
        self.stream = cv2.VideoCapture(src)
        
        # fourcc = cv2.VideoWriter_fourcc('M','J', 'P', 'G')  # read compressed stream from camera
        # self.stream.set(cv2.CAP_PROP_FOURCC, fourcc)

        self.stream.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.stream.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)
        self.stream.set(cv2.CAP_PROP_FPS, 30)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        #keep looping infinitely untill the thread is stopped
        while True:
            #if the thread indicator variable is set, stop the thread
            if self.stopped:
                return
            #otherwise read the next frame from the stream
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        #return the frame most recently read
        return (self.grabbed, self.frame)

    def stop(self):
        #indicate that the thread should be stopped
        self.stopped = True
