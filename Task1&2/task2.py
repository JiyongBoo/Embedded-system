import os
from jetracer.nvidia_racecar import NvidiaRacecar
from datetime import datetime, timedelta  # don't forget to import datetime
from move1 import PIDController
import atexit
import time

import argparse
args = argparse.ArgumentParser()
args.add_argument('--stream',
        action = 'store_true',
        default= False,
        help = 'Launch OpenCV window and show livestream')
args = args.parse_args()
stream = args.stream


car = NvidiaRacecar()

import pygame
from camera import *
from ultralytics import YOLO

def stop_driving():
    car.steering = 0.0
    car.throttle = 0.0
    cam.cap[0].release()

atexit.register(stop_driving) # 터미널 시그널로 종료시 수행

def detectline(image): # 차선 인지 하여 binary img 반환
    roi_image = image[200:,:,:]

    # hls로 변환, yellow 임계값 필터로 차선 부분 필터링
    hls = cv2.cvtColor(roi_image, cv2.COLOR_BGR2HLS)

    # color filtering
    yellow_lower = np.array([0, 130, 30]) # Hyper parameter
    yellow_upper = np.array([85, 250, 255]) # Hyper parameter

    yellow_mask = cv2.inRange(hls, yellow_lower, yellow_upper)
    masked = cv2.bitwise_and(roi_image, roi_image, mask = yellow_mask)

    # gray scale로 변환 후 gaussian filtering
    masked_gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)
    masked_gray = cv2.GaussianBlur(masked_gray, (0, 0), 3)

    # binary img로 변환
    _, binary_img = cv2.threshold(masked_gray, 140, 255, cv2.THRESH_BINARY)

    return binary_img

def find_points(x_list, y_list, cls):
    # y 좌표가 가장 큰 점 a 찾기
    max_y = max(y_list)
    index_a0 = y_list.index(max_y)
    a0 = (x_list[index_a0], y_list[index_a0])

    find_a = [(x_list[i], y_list[i]) for i in range(len(x_list)) 
                    if abs(y_list[i] - a0[1]) <= 30]
    
    if cls == 'left':
        a = min(find_a, key=lambda point: point[0])
        index_a =  y_list.index(a[1]) 
        close_points = [(x_list[i], y_list[i]) for i in range(len(x_list)) 
                    if -330 < (x_list[i] - a[0]) <= 200 and abs(y_list[i] - a[1]) <= 130 and i != index_a]
    elif cls == 'right':
        a = max(find_a, key=lambda point: point[0])
        index_a =  y_list.index(a[1]) 
        close_points = [(x_list[i], y_list[i]) for i in range(len(x_list)) 
                    if -200 < (x_list[i] - a[0]) <= 330 and abs(y_list[i] - a[1]) <= 130 and i != index_a]
    elif cls == 'straight':
        a = min(find_a, key=lambda point: abs(point[0] - 480))
        index_a =  y_list.index(a[1]) 
        close_points = [(x_list[i], y_list[i]) for i in range(len(x_list)) 
                    if abs(x_list[i] - a[0]) <= 50 and abs(y_list[i] - a[1]) <= 200 and i != index_a]
    else:
        a = a0
        index_a =  y_list.index(a[1]) 
        close_points = [(x_list[i], y_list[i]) for i in range(len(x_list)) 
                    if -330 < (x_list[i] - a[0]) <= 200 and abs(y_list[i] - a[1]) <= 130 and i != index_a]
    



    # a와 가장 가까운 점 b 찾기
    if close_points:
        if cls == 'left':
            b = min(close_points, key=lambda point: point[0])
        elif cls == 'right':
            b = max(close_points, key=lambda point: point[0])
        elif cls == 'straight':
            b = min(close_points, key=lambda point: ((point[0] - a[0])**2 + (point[1] - a[1])**2)**0.5)
        else:
            # b = min(close_points, key=lambda point: point[0]) 
            b = min(close_points, key=lambda point: point[0])
    else:
        b = None  # 가까운 점이 없을 경우 None 반환

    return a, b

cam  = Camera(sensor_id = 0)

model = YOLO("best2.engine", task='detect')
classes = model.names
cam.set_model(model, classes) # bus, crosswalk, right, left, straight

# Initialize PID controller
# Using Ziegler-Nichols method
Tu = 0.75
Ku = 2.5
Kp_s = 2.0 * Tu
Ki_s = 0.0
Kd_s = 0.0
Kp_d = 1.5 # Kp_d = 0.6 * Ku    # opt value: 1.5
Ki_d = 3.0 # Ki_d = 1.2 * Ku / Tu    # opt value: 3.0
Kd_d = 0.1875 # Kd_d = 0.075 * Ku * Tu    # opt value: 0.1875
alpha = 0.5     # weight for steering
beta = 0.5      # weight for throttle
steering_limits = (-1.0, 1.0)  # Steering limits,  throttle_range = (-0.4, 0.4)
throttle_limits = [0.184,0.182]  # Operating area: 전진 0.16, 후진 0.178

slopePID = PIDController(Kp_s, Ki_s, Kd_s, throttle_limits=throttle_limits)
difPID = PIDController(Kp_d, Ki_d, Kd_d, setpoint = -0.125, throttle_limits=throttle_limits)


# For headless mode
os.environ["SDL_VIDEODRIVER"] = "dummy"

pygame.init()
pygame.joystick.init()

joystick = pygame.joystick.Joystick(0)
joystick.init()


running = True # While 문 flag
capture = False # 사진 저장 flag

pred_count = 9 # object detection count 변수 -> 10프레임 당 한번만 detect

bus_detect_time = 0 
cross_detect_time = None
stop_time = None
direction_cls = None
cls = None
box_width = 0

button0 = False
button7 = False
button9 = False

while running:
    pygame.event.pump()

    # 버튼 조작
    if joystick.get_button(11): # start button
        running = False
        capture = False

    if joystick.get_button(0) == True and joystick.get_button(0) != button0: # 0번 버튼 누를시 capture start / finish
        capture = not capture
    button0 = joystick.get_button(0)
    
    if joystick.get_button(7) and joystick.get_button(7) != button7: # 7번 버튼 누를시 throttle limit 증가
        throttle_limits[0] += 0.002
        throttle_limits[1] += 0.002
    button7 = joystick.get_button(7)

    if joystick.get_button(9) and joystick.get_button(9) != button9: # 9번 버튼 누를시 throttle limit 감소
        throttle_limits[0] -= 0.002
        throttle_limits[1] -= 0.002
    button9 = joystick.get_button(9)

    # 이미지 받아들이기   
    t0 = time.time()
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S%f')
    _, frame = cam.cap[0].read()

    # YOLO model
    pred_count += 1
    if cam.model is not None and pred_count > 10:
        # run model
        pred = cam.model.predict(frame, stream=True, conf = 0.5, half = True)
        # visualize prediction
        size, cls, box_width = draw_boxes(frame, pred, cam.classes, cam.colors)
        pred_count = 0
    
    # 방향 모드 선택
    if cls == 'straight' or cls == 'left' or cls == 'right':
        direction_cls = cls
    elif cls == 'bus' or cls == 'crosswalk':
        direction_cls = None

    # 차선 검출
    line= detectline(frame)
    contours, hierarchy = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,offset=[0,200])
    try: 
        max_y = 0
        x ,y = [],[]
        for i in range(len(hierarchy[0,:,0])):
            x.append(round(np.mean(contours[i][:,0,0])))
            y.append(round(np.mean(contours[i][:,0,1])))
            max_y = max(y[i],max_y)
        for i in range(len(y)):
            cv2.line(frame, (x[i],y[i]), (x[i],y[i]), color=(0,0,255),thickness=5) 
            
        # 점 개수로 교차로 진입 확인    
        if len(x) > 13:
            ICflag = True
        else:
            ICflag = False

        a, b = find_points(x,y,cls=direction_cls) # Resolution(x, y): (960, 540)

        if b is not None:
            current_slope = (b[0] - a[0]) / (b[1] - a[1]) # left(+), right(-)
        else:
            current_slope = 0

        current_dif = (480 - a[0]) / 480


        # 차량 제어
        # 버스 전용 차로 감지 확인
        if cls == 'bus' and size > 1000:  # bus sign 보일 때 + 4초까지만 동작(표지판 사라지고 4초까지만)
            bus_detect_time= time.time()

        # 횡단보도 감지 확인 후 정지    
        elif cls == 'crosswalk' and cross_detect_time is None and size > 10000:
            cross_detect_time = time.time()
            stop_time = time.time()
            slopePID.throttle_limits = [0,0]
            difPID.throttle_limits = [0,0]

        # 버스 전용 차로 감지 확인 시, 보이지 않게 된 후 2초 까지 감속
        if time.time() - bus_detect_time < 2 :
            slopePID.throttle_limits = [throttle_limits[0]-0.006,throttle_limits[1]-0.006]
            difPID.throttle_limits = [throttle_limits[0]-0.006,throttle_limits[1]-0.006]
            difPID.update(current_dif,stopflag = False, ICflag=ICflag)

        # 횡단 보도 최초 감지 확인 후 4초간 정지,       
        elif stop_time and (time.time()-stop_time < 4):
            slopePID.throttle_limits = [0,0]
            difPID.throttle_limits = [0,0]
            difPID.update(current_dif,stopflag=True,ICflag=ICflag)

        # 감속/정지 이후 서서히 정상 속도로 가속    
        else:
            if slopePID.throttle_limits[0] < throttle_limits[0]:
                slopePID.throttle_limits[0] += 0.002
                slopePID.throttle_limits[1] += 0.002
                difPID.throttle_limits[0] += 0.002
                difPID.throttle_limits[1] += 0.002
            else:
                slopePID.throttle_limits = throttle_limits
                difPID.throttle_limits = throttle_limits
            stop_time = None
            difPID.update(current_dif,stopflag = False,ICflag=ICflag)

        # 횡단보도 최초 감지 이후 10초간 감지 무시
        if cross_detect_time and (time.time()-cross_detect_time) > 10:
            cross_detect_time = None
        
        # 기울기 정보에 따른 제어
        if -0.8 <= current_slope < -0.6: # right 30 ~ 40 degree      
            slopePID.update(current_slope)
            car.steering = alpha * slopePID.steering + (1-alpha) * difPID.steering
            car.throttle = beta * slopePID.throttle + (1-beta) * difPID.throttle
        elif 0.8 < current_slope <= 1.2: # left 40 ~ 50 degree      
            slopePID.update(current_slope)
            car.steering = alpha * slopePID.steering + (1-alpha) * difPID.steering
            car.throttle = beta * slopePID.throttle + (1-beta) * difPID.throttle    
        elif current_slope < -0.8: # right over 40 degree  *lower*
            slopePID.update(current_slope)
            car.steering = slopePID.steering
            car.throttle = slopePID.throttle
        elif current_slope > 1.2: # left over 50 degree
            slopePID.update(current_slope)
            car.steering = slopePID.steering
            car.throttle = slopePID.throttle
        else:   # when only 1 point is detected
            slopePID.update(0)
            current_dif = (480 - a[0]) / 480 
            car.steering = difPID.steering
            car.throttle = difPID.throttle
        
        # 주행 정보 시각화
        cv2.circle(frame, a, 5, color=(0,0,255),thickness=2)
        cv2.putText(frame, f'{a}', a, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.circle(frame, b, 5, color=(255,0,0),thickness=2)
        cv2.putText(frame, f'{b}', b, cv2.FONT_HERSHEY_SIMPLEX, 0.8,
            (255, 0, 0), 1, cv2.LINE_AA)
        if b is not None:
            cv2.line(frame,a,b,color = (0,255,0),thickness=2)
        
        cv2.putText(frame, f'thr: {car.throttle}', (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'str: {car.steering}', (0,60), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'sl_str: {slopePID.steering}', (0,90), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'dif_str: {difPID.steering}', (0,120), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'cls: {cls}', (0,150), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'size: {size}', (0,180), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'dir: {direction_cls}', (0,210), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
            (0, 0, 255), 1, cv2.LINE_AA)
    except:
        pass



    # args로 --stream 받을 시 실시간 시각화(ssh로 실행할 시 오류)
    if stream:
        cv2.namedWindow(cam.window_title)
        cv2.imshow(cam.window_title, frame)
        cv2.waitKey(1)
	
    # 0번 버튼 누를 시 주행 영상 저장
    if capture ==  True:      
        cv2.imwrite(str(cam.save_path / f"{timestamp}.jpg"), frame)
        print('saving : ',str(cam.save_path / f"{timestamp}.jpg"))
    
    # print("throttle: {:f}, steering: {:f}".format(car.throttle, car.steering))
    # print("slope_throttle: {:f}, slope_steering: {:f}".format(slopePID.throttle, slopePID.steering))
    # print("dif_throttle: {:f}, dif_steering: {:f}".format(difPID.throttle, difPID.steering))
    # # print(a,b)
    # print("current_slope, current_dif", current_slope, current_dif)


cam.cap[0].release()
cv2.destroyAllWindows()
stop_driving()