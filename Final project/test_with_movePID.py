from __future__ import annotations
from pathlib import Path
from typing import Sequence

import argparse
import cv2
import datetime
import glob
import logging
import numpy as np
import os
import time
from ultralytics import YOLO

from jetracer.nvidia_racecar import NvidiaRacecar
from move2 import *
import pygame
import atexit
from test import *
import gc

gc.collect()

def stop_driving():
    car.steering = 0.0
    car.throttle = 0.0
    cam.cap[0].release()
    print("stop driving")

atexit.register(stop_driving)   # stop car and camera at exit



# from jtop import jtop # Use this to monitor compute usage (for Jetson Nano)

logging.getLogger().setLevel(logging.INFO)

def which_point_to_follow(points, d_points, desired_point_y:tuple):
    # TODO: this method only use two points. Find way to use spline.
    # desired_point_y = (close_point_y, far_point_y)
    if any(y <= 240 for _, y in points):
        stopflag = False
        for p in points:
            if p[1] == desired_point_y[0]:
                dif0 = p[0]
                dif0 = (455 - dif0) / 455
            elif p[1] == desired_point_y[1]:
                dif1 = p[0]
                dif1 = (455 - dif1) / 455
        for dp in d_points:
            if dp[1] == desired_point_y[0]:
                slope0 = dp[0]
            elif dp[1] == desired_point_y[1]:
                slope1 = dp[0]
        # if (dif is None or slope is None):
        #     print("dif is None or slope is None")
        #     pass

    else:
        stopflag = True
        dif0 = None
        dif1 = None
        slope0 = None
        slope1 = None

    return dif0, dif1, slope0, slope1, stopflag
    


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--sensor_id',
        type = int,
        default = 0,
        help = 'Camera ID')
    args.add_argument('--window_title',
        type = str,
        default = 'Camera',
        help = 'OpenCV window title')
    args.add_argument('--save_path',
        type = str,
        default = 'record',
        help = 'Image save path')
    args.add_argument('--save',
        action = 'store_true',
        help = 'Save frames to save_path')
    args.add_argument('--stream',
        action = 'store_true',
        help = 'Launch OpenCV window and show livestream')
    args.add_argument('--log',
        action = 'store_true',
        help = 'Print current FPS')
    args = args.parse_args()

    cam = Camera(
        sensor_id = args.sensor_id,
        window_title = args.window_title,
        save_path = args.save_path,
        save = args.save,
        stream = args.stream,
        log = args.log)
        
    model = YOLO("./roadseg.engine", task = 'segment')
    cam.model = model
    # img = cv2.imread('/home/ircv/HYU-2024-Embedded/team2/run/record/000066/20240607100228745708.jpg')
    # results = model(img)
    # print(results[0])
    Cameramtx = np.load('Cameramtx.npz')
    mtx = Cameramtx['mtx']
    dist = Cameramtx['dist']
    newcameramtx, _ = cv2.getOptimalNewCameraMatrix(mtx,dist,(960,540),0)
    mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(960,540),5)

    car = NvidiaRacecar()

    # PID
    Kp_s = 2.0  # 2.0*0.75
    Ki_s = 0.0
    Kd_s = 0.0
    Kp_d = 2.0
    Ki_d = 0.5
    Kd_d = 0.0
    alpha = 0.0 # slope에 가중치

    desired_point_y = (320, 260)  # (320, 280)

    desired_slope = 0.0
    desired_dif = -0.0044    # (455-457)/455

    slopeSteering = SteeringController(Kp_s, Ki_s, Kd_s, setpoint = desired_slope)
    difSteering = SteeringController(Kp_d, Ki_d, Kd_d, setpoint = desired_dif)
    throttle = ThrottleController(tolerances_a=(3,3), tolerances_b=(10,10), throttle_gain = 0.01, throttle_decrease_rate = 1, throttle_limits=(0, 0.195))
    following_dif = 0.0
    following_slope = 0.0

    # For using joystick
    os.environ["SDL_VIDEODRIVER"] = "dummy"

    pygame.init()
    pygame.joystick.init()

    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    button0 = False

    start_count = 0

    running = True
    while running:
        start_count +=1
        # Below is the endless loop while driving
        
        pygame.event.pump()
        
        if joystick.get_button(11): # start button
            running = False
            capture = False
        elif (joystick.get_button(0) == True) and (joystick.get_button(0) != button0):
            capture = not capture
        button0 = joystick.get_button(0)
        
        t0 = time.time()
        timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
        _, frame = cam.cap[0].read()

        
        try:
            frame = cv2.remap(frame,mapx,mapy,cv2.INTER_LINEAR)
            center_points, poly_points, d_poly_points, pre_frame, _ = draw_center_line2(cam.model,frame) # center_points: y 20마다 x 중간값, poly_points: 보간된 선.
            # now we can use pre_frame as colored image, and center_points as desired following line
            if start_count > 300:

            
                following_dif, dif_far, _, following_slope, stopflag = which_point_to_follow(poly_points, d_poly_points, desired_point_y)
                if (abs(poly_points[0][0] - poly_points[-1][0]) > 250):
                    # corner
                    alpha = 1.0 # use slope
                else:
                    # straight
                    alpha = 0.0 # use dif


                if stopflag:
                    throttle.update(throttle_reference = 0.0, features = poly_points[-1])
                    slopeSteering.update(None)
                    difSteering.update(None)
                    # car.throttle = 0.0
                else:
                    throttle.update(throttle_reference = 0.16, features = poly_points[-1])
                    slopeSteering.update(following_slope)
                    difSteering.update(following_dif)
                    # car.throttle = 0.195
                car.throttle = throttle.throttle
                car.steering = alpha * slopeSteering.steering + (1-alpha) * difSteering.steering - 0.296 # steering offset for ircv5
                # car.steering = (car.steering ** 2) * (1 if car.steering > 0 else -1)
                # car.steering = - (10.0 * following_dif + 0.0 * following_slope) - 0.296

                
                
                # TODO: use joystick button to save images
                cv2.putText(pre_frame, f'thr: {car.throttle}', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(pre_frame, f'str: {car.steering+0.296}', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(pre_frame, f'sl_str: {slopeSteering.steering}', (0,90), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(pre_frame, f'dif_str: {difSteering.steering}', (0,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(pre_frame, f'fol_dif: {following_dif}', (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                cv2.putText(pre_frame, f'fol_slp: {following_slope}', (0,180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 1, cv2.LINE_AA)
                if (following_dif is not None and following_slope is not None and dif_far is not None):
                    cv2.line(pre_frame, (round(455*(1-following_dif)), round(desired_point_y[0])), (round(455*(1-following_dif)), round(desired_point_y[0])), color=(0,0,255), thickness=5)
                    cv2.line(pre_frame, (round(455*(1-dif_far)), round(desired_point_y[1])), (round(455*(1-dif_far)), round(desired_point_y[1])), color=(0,0,255), thickness=5)

                if args.stream:
                    cv2.namedWindow(cam.window_title)
                    cv2.imshow(cam.window_title, pre_frame)
                    cv2.waitKey(1)
            else:
                car.steering = 0.0
                car.throttle = 0.0
            if args.save:
                cv2.imwrite(str(cam.save_path / f"{timestamp}.jpg"), pre_frame)
                print('saving : ',str(cam.save_path / f"{timestamp}.jpg"))

        except:
            pass
