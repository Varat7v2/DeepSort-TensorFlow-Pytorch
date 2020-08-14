# file: videocaptureasync.py
import threading
import cv2
from time import sleep
import copy

class VideoCaptureAsync:
    def __init__(self, width=2688, height=1520, thermal=True):
        # self.src = 'rtsp://admin:admin123@192.168.1.108:554'
        # #self.src = 'rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=1'
        # #sirve frame pequeño
        # self.src = 'rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0'
        # #frame grande
        # if (thermal):
        #     self.src = 'rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=2&subtype=0'
        # else:
        #     self.src = 'rtsp://admin:admin123@192.168.1.108:554/cam/realmonitor?channel=1&subtype=0'
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.grabbed, self.frame = self.cap.read()
        self.started = False
        self.read_lock = threading.Lock()

    def set(self, var1, var2):
        self.cap.set(var1, var2)

    def start(self):
        if self.started:
            print('[!] Asynchroneous video capturing has already been started.')
            return None
        self.started = True
        self.thread = threading.Thread(target=self.update, args=())
        self.thread.start()
        return self

    def update(self):
        while self.started:
            sleep(0.03)
            grabbed, frame = self.cap.read()
            with self.read_lock:
                self.grabbed = grabbed
                self.frame = frame

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def isOpened(self):
        return self.cap.isOpened()

    def stop(self):
        self.started = False
        self.thread.join()
    def release(self):
        self.cap.release()

    def __exit__(self, exec_type, exc_value, traceback):
        self.cap.release()