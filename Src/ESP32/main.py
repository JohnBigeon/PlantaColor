# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 12:48:45 2024

@author: JBI
"""


"""
###############################################
##Title             : plantacolor_esp32_main.py
##Description       : Main script for LED control
##Author            : John Bigeon   @ Github
##Date              : 20230730
##Version           : Test with
##Usage             : MicroPython (esp32-20240222-v1.12.2)
##Script_version    : 0.0.1 (not_release)
##Output            :
##Notes             :
###############################################
"""
###############################################
### Package
###############################################
import select
import sys
import machine
import utime
import re
import machine, neopixel

###############################################
### Params
###############################################
num_led_max = 12
quarter = num_led_max //4

### Init devices
np = neopixel.NeoPixel(machine.Pin(9), num_led_max)

###############################################
### MAIN
###############################################
num_iter = 0
while True:
    if select.select([sys.stdin],[],[],0)[0]:
        ch = sys.stdin.readline()
        #print(ch)
        try:
            ch = ch.decode('utf-8')
        except:
            pass
        #ch = b'Mode req: Q=1; R=32; G=230; B=139\n'

        # Regular expression pattern to match the values of R, G, and B
        pattern = r'Q=(\d+); R=(\d+); G=(\d+); B=(\d+)'

        match = re.search(pattern, ch)  # Using re.search() to find the pattern in the string since re.findall() is not implemented in python
        
        if match:
            # Extracted values
            q_val = int(match.group(1))
            r_val = int(match.group(2))
            g_val = int(match.group(3))
            b_val = int(match.group(4))
            
            for ii in range(num_led_max):
                np[ii] = (0,0,0)
            
            for ii in range(quarter):
                np[q_val*quarter + ii] = (r_val, g_val, b_val)
                
            np.write()
            utime.sleep(0.05)
            print('OK')
            # Reset timeout counter on successful input processing
            num_iter = 0
        else:
            print('E02: Not decrypted')

    elif num_iter > 2000:  # Timeout condition after 100 seconds
        for ii in range(num_led_max):
            np[ii] = (255, 255, 255)  # Set LEDs to white
        np.write()
        utime.sleep(0.05)
    else:
        num_iter += 1
        utime.sleep(0.05)
