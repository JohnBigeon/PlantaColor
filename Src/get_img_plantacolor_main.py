# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 18:06:01 2024

@author: JBI
"""
"""
###############################################
##Title             : get_img_plantacolor_main.py
##Description       : Main script for Plantacolor project
##Author            : John Bigeon   @ Github
##Date              : 20240316
##Version           : Test with
##Usage             : MicroPython (esp32-20220618-v1.19.1)
##Script_version    : 0.0.5 (not_release)
##Output            :
##Notes             : Remove matplotlib warning in logging, contour sum them !
###############################################
"""
###############################################
### Package
###############################################
import numpy as np
import serial
import time
import cv2
import threading
import sys
import os
import logging
import datetime
import matplotlib.pyplot as plt
import Lib_local as mylib
import Lib_local.video_webcam as cam


###############################################
### Versioning
def versioning():
    ver = "# Plantacolor: {mylib.__version__}"
    return ver

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
datum = datetime.datetime.now()
log_file = os.path.join(log_dir, f"logfile_plantacolor_{datum:%Y-%m-%d_%H%M%S}.log")
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


def wavelength_to_rgb(wavelength):
    # Based on the provided Bruton's Algorithm
    # https://stackoverflow.com/questions/3407942/rgb-values-of-visible-spectrum
    assert 350 <= wavelength <= 780
    if wavelength < 440:
        R = (440 - wavelength) / (440 - 350)
        G = 0
        B = 1
    elif wavelength < 490:
        R = 0
        G = (wavelength - 440) / (490 - 440)
        B = 1
    elif wavelength < 510:
        R = 0
        G = 1
        B = (510 - wavelength) / (510 - 490)
    elif wavelength < 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1
        B = 0
    elif wavelength < 645:
        R = 1
        G = (645 - wavelength) / (645 - 580)
        B = 0
    else:
        R = 1
        G = 0
        B = 0

    # Adjust intensity
    if wavelength > 700:
        intensity = 0.3 + 0.7 * (780 - wavelength) / (780 - 700)
    elif wavelength < 420:
        intensity = 0.3 + 0.7 * (wavelength - 350) / (420 - 350)
    else:
        intensity = 1

    R = (R * intensity) * 255
    G = (G * intensity) * 255
    B = (B * intensity) * 255

    return np.array([R, G, B], dtype=int)


def stepping_wvl(video_flux, ser, datum):
        # Prepare data
    exp_path = "Data//Data_{0:%Y-%m-%d_%H%M%S}".format(datum)
    os.makedirs(exp_path, exist_ok=True)

    # Init vectors
    qval_range = np.arange(0, 4)
    wvl_var = np.arange(350, 780, 15)

    cval_range = []
    for _, wvl_tmp in enumerate(wvl_var):
        cval_range.append(wavelength_to_rgb(wvl_tmp))
        print(wvl_tmp, wavelength_to_rgb(wvl_tmp))

    iter_num_max = len(cval_range) * len(qval_range)
    iter_num = 0

    # Generate logbook
    # Create a file handler for the additional log file
    dat_log_file = "Data//logfile_stepping_{0:%Y-%m-%d_%H%M%S}.log".format(datum)
    dat_file_handler = logging.FileHandler(dat_log_file, mode="w")
    dat_file_handler.setLevel(logging.INFO)
    dat_file_handler.setFormatter(formatter)

    # Log to the additional log file
    dat_logger = logging.getLogger("dat_logger")
    dat_logger.addHandler(dat_file_handler)
    dat_logger.info('Filename; Wvl; Qval; Rval; Gval; Bval')

    # Main iteration
    for _, wvl_tmp in enumerate(wvl_var):
        rval_tmp, gval_tmp, bval_tmp = wavelength_to_rgb(wvl_tmp)
        for q_val_idx, qval_tmp in enumerate(qval_range):

            print(f'##################### Iter: {iter_num}/{iter_num_max-1} #####################')
            logging.info(f'Illumination: wvl={wvl_tmp} Q={qval_tmp}; R={rval_tmp}; G={gval_tmp}, B={bval_tmp}')

            command = f'Mode req: Q={qval_tmp}; R={rval_tmp}; G={gval_tmp}; B={bval_tmp}\n'
            cmd = bytes(command.encode('ascii'))
            req_done = False

            while req_done is False:
                logging.info('Request to the device %s' %cmd)

                ser.write(cmd)

                if ser.inWaiting() > 0:
                    ser_data = ser.readline()
                    logging.info(f'Msg received: {ser_data}')

                    ser_data = ser_data.decode("utf-8","ignore")
                    print(5*'#')
                    time.sleep(0.05)

                    if 'OK' in ser_data:
                        time.sleep(0.25)
                        img = video_flux.frame # Grab one image
                        datum = datetime.datetime.now().strftime("%Y%m%d_%H%M%S.%f")

                        snapshot_filename = exp_path + '/' + f'img_{datum}_qval{qval_tmp}_rval{rval_tmp}_gval{gval_tmp}_bval{bval_tmp}'
                        snapshot_filename += '.bmp'

                        cv2.imwrite(snapshot_filename, img)

                        # Calculate the percentage of pixels above 200
                        threshold = 254
                        num_above_threshold = (img > threshold).sum()
                        percent_above_threshold = 100 * num_above_threshold / img.size

                        if percent_above_threshold > 10:
                            logging.warning('Image acquired is satured at 10%')


                        dat_logger.info(f"{snapshot_filename}; {wvl_tmp}; {qval_tmp}; {rval_tmp}; {gval_tmp}; {bval_tmp}")

                        time.sleep(0.05)
                        req_done = True
                        iter_num += 1

                    else:
                        logging.error(ser_data)
                        time.sleep(0.05)

                time.sleep(0.05)

        # Remove the handler after using it
    dat_logger.removeHandler(dat_file_handler)

    # Turn off last led selection
    command = f'Mode req: Q={qval_tmp}; R=0; G=0; B=0\n'
    cmd = bytes(command.encode('ascii'))
    logging.info('Request to the device %s' %cmd)
    ser.write(cmd)
    time.sleep(0.1)     # Wait to let the last message arrived to the device


def lux_generation(ser, qval=2, rval=255, gval=255, bval=255):

    command = f'Mode req: Q={qval}; R={rval}; G={gval}; B={bval}\n'
    cmd = bytes(command.encode('ascii'))
    req_done = False
                    
    while req_done is False:
        logging.info('Request to the device %s' %cmd)

        ser.write(cmd)
        
        if ser.inWaiting() > 0:
            ser_data = ser.readline()
            logging.info(f'Msg received: {ser_data}')

            ser_data = ser_data.decode("utf-8","ignore")
            print(5*'#')
            
            if not 'OK' in ser_data:
                sys.exit()
            else:
                req_done = True
                time.sleep(0.05)
        else:
            time.sleep(0.05)
                


def video_desaturation(video_flux):
    img_saturated = True # Init to go in the test procedure
    exposure_val = 0
    # Calculate the percentage of pixels above 254
    threshold = 254    
    
    img_test = None
    while img_test is None:
        if img_saturated == True:
            if hasattr(video_flux, 'frame'):
                video_flux.set_exp_val(exposure_val) # By default, set exposure time to zero
                time.sleep(0.1)
                while img_saturated:
                    time.sleep(2.0)
                    img_test = video_flux.frame # Grab one image
                    img_test = cv2.cvtColor(img_test, cv2.COLOR_BGR2GRAY)

                    num_above_threshold = (img_test > threshold).sum()
                    percent_above_threshold = 100 * num_above_threshold / img_test.size
                    
                    plt.imshow(img_test)
                    plt.title(f'Exposure val: {exposure_val}, Saturated ?')
                    plt.colorbar()
                    plt.pause(2.0)
                    plt.close()
                                        
                    if percent_above_threshold > 10:
                        logging.warning('Image acquired is satured at 10%')
                        if exposure_val < -13:
                            logging.error('Cannot reduce the exposure time here')
                        else:
                            exposure_val -= 1
                            video_flux.set_exp_val(exposure_val)
                            time.sleep(1.0)
                        
                    else:
                        img_saturated = False
                        del(video_flux.frame)

        else:
            time.sleep(0.5)
            logging.info('Webcam: still warmup')
            
    logging.info(f'Exposure time of the camera should be fixed at {exposure_val}')
    exposure_req = input(f'Please enter the exposure required\n')
    exposure_req = int(exposure_req)
    video_flux.set_exp_val(exposure_req)
    logging.info(f'Exposure time hardcoded to {exposure_req}')

###############################################
### Main
###############################################
def main(com_port='COM13'):   
    # Inspiration:     #https://forums.raspberrypi.com/viewtopic.php?f=144&t=301414#p1809106

    # Timing
    time_start = time.time()
    datum = datetime.datetime.now()    

    # Init esp32 device
    logging.info("Controller: Init.")    
    ser = serial.Serial(com_port, 115200)
    
    # Init camera
    logging.info("Webcam: Init.")    
    video_flux = cam.Video(1)
    video_flux.start_streaming()

    lux_generation(ser, qval=2, rval=255, gval=255, bval=255)

    video_desaturation(video_flux)
    time.sleep(0.25)

    # Test live image
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)

    while True:
        # Display the frame if it has been updated
        if hasattr(video_flux, 'frame'):
            cv2.imshow('image', video_flux.frame)
            cv2.resizeWindow('image', 500, 500)
            time.sleep(0.02)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    time.sleep(0.5)
    cv2.destroyAllWindows()
    time.sleep(1.0)

    stepping_wvl(video_flux, ser, datum)
    time.sleep(0.25)
    
    # Close devices
    video_flux.close()
    ser.close()
    
    elapsed_time = time.time() - time_start
    logging.info(f'Elapsed time: {elapsed_time} seconds')

###############################################
### Main
###############################################
if __name__ == "__main__":
    main(com_port='COM13')    
