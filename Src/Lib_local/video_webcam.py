# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:06:01 2024

@author: JBI
"""
"""
###############################################
##Title             : video_webcam.py
##Description       : Webcam video for plantacolor project
##Author            : John Bigeon   @ Github
##Date              : 20240316
##Version           : Test with
##Usage             : Python 3.10.12
##Script_version    : 0.0.1 (not_release)
##Output            :
##Notes             : 
###############################################
"""
###############################################
### Package
###############################################
import numpy as np
import os
import datetime
import time
import cv2
import logging
import threading


###############################################
### Logger function
###############################################
# Configure logging with dynamic log file path
# Create the log directory if it doesn't exist
log_dir = "Log"
os.makedirs(log_dir, exist_ok=True)

# Configure the root logger
logging.basicConfig(level=logging.DEBUG)

# Create a file handler for writing to the log file
log_file = os.path.join(log_dir, "logfile_webcam_{0:%Y-%m-%d_%H%M%S}.log".format(datetime.datetime.now()))
file_handler = logging.FileHandler(log_file, mode="w")
file_handler.setLevel(logging.DEBUG)

# Create a stream handler for printing to the terminal
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)

# Define the log message format
formatter = logging.Formatter("%(asctime)s; %(levelname)s; %(message)s")

# Set the formatter for both handlers
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)

# Get the root logger and clear any existing handlers
root_logger = logging.getLogger()
root_logger.handlers.clear()

# Add the handlers to the root logger
root_logger.addHandler(file_handler)
root_logger.addHandler(stream_handler)

###############################################
### Video Class for webcam Camera
###############################################
class Video():

    def __init__(
        self,
        camera_port=0,
        parent=None,
    ):
        logging.info("Camera: init.")
        self.standby = True
        self.camera_port = camera_port

    def run(self):
        # Initialize the camera capture object
        self.cap = cv2.VideoCapture(self.camera_port)

        # Set the camera capture properties
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)

        # Continuously read frames from the camera
        while self.standby:
            ret, frame = self.cap.read()
            if ret:
                self.frame = frame

    def capture(self):
        return self.frame

    # Launches the video recording function without a thread
    def start_streaming(self):
        # Start a thread to continuously read frames
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True  # Daemonize the thread so it will be terminated when the main program exits
        self.thread.start()
        
    def set_exp_val(self, exposure_val: int):
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_val)
        logging.info('Camera: Exposure val at %d', exposure_val)


    def set_exp(self, exposure_time):

        # Table according to: https://www.kurokesu.com/main/2020/05/22/uvc-camera-exposure-timing-in-opencv/
        exposure_values = {
            1.0: 0,
            0.5: -1,
            0.25: -2,
            0.125: -3,
            0.0625: -4,
            0.03125: -5,
            0.015625: -6,
            0.0078125: -7,
            0.00390625: -8,
            0.002: -9,
            0.0009766: -10,
            0.0004883: -11,
            0.0002441: -12,
            0.0001221: -13
        }
        closest_value = min(exposure_values, key=lambda x: abs(x - exposure_time))
        self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure_values[closest_value])

    def stop_streaming(self):
        # Stop the streaming thread
        self.standby = False
        self.thread.join()  # Wait for the thread to terminate
        self.cap.release()  # Release the camera capture object

    def close(self):
        self.standby = False
        cv2.destroyAllWindows()
        logging.info("Camera: quit")

    def version(self):
        ver = "# Webcam : 0.0.1"
        return ver