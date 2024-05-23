from datetime import datetime, timedelta  # don't forget to import datetime

class PIDController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, steering_limits=(-1.0, 1.0),throttle_limits=(0.19, 0.20)):
 
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0         # cumulative error
        self.pre_error = 0.0        # previous error
        self.pre_t = None           # previous time
        self.setpoint = setpoint    # target setpoint, 가운데가 0이 아닐수도 있음. Offset 조절 필요할지도
        self.steering_limits = steering_limits  # (min_output, max_output)
        self.throttle_limits = throttle_limits
        self.steering = 0.0
        self.throttle = 0.0
        self.Ka = 2.0/Kp        # Anti-Windup coefficient
        self.anti = 0.0

    
    def update(self, output, stopflag=False, ICflag = False):
        # output must be "current value from sensor", not the actual output of the function
        # The actual output of the function is "self.steering" & "self.throttle". You should call this variable to use for car.steering & car.throttle
        error = self.setpoint - output 

        cur_t = datetime.now()  # needs to import `datetime`
        if self.pre_t is None:
            dt = 0.0
        else:
            dt = (cur_t - self.pre_t).total_seconds()
            print(dt)

        self.pre_t = cur_t

        if dt > 0.0:
            # self.integral += error * dt   # without anti-windup
            if stopflag: # 신호등 확인으로 정지 시
                derivative = 0
            elif ICflag: # 교차로 진입 시
                self.integral += (error - self.Ka * self.anti) * dt
                derivative = 0
            else:
                self.integral += (error - self.Ka * self.anti) * dt
                derivative = (error - self.pre_error) / dt
        else:
            # when dt == 0, prevent dividing zero
            derivative = 0.0

        self.pre_error = error

        self.steering = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        

        # Anti-windup: Clamp the integral term if control is beyond limits
        # if (self.steering_limits[0] is not None) and (self.steering < self.steering_limits[0]):
        #     self.steering = self.steering_limits[0]
        #     if error < 0:  # Prevent further negative integral accumulation
        #         self.integral -= error * dt # error * dt < 0, increase integral
        # elif (self.steering_limits[1] is not None) and (self.steering > self.steering_limits[1]):
        #     self.steering = self.steering_limits[1]
        #     if error > 0:  # Prevent further positive integral accumulation
        #         self.integral -= error * dt # error * dt > 0, decrease integral
        if self.steering > self.steering_limits[1]:
            saturated = self.steering_limits[1]
        elif self.steering < self.steering_limits[0]:
            saturated = self.steering_limits[0]
        else:
            saturated = self.steering
        self.anti = self.steering - saturated
        
        self.steering = saturated
        

        # 표지판 검출되면 속도 변화 (ex: 버스전용차로 5초간 스로틀 0.17고정, 횡단보도 3초간 정지)

        max_throttle = self.throttle_limits[1]  # Maximum throttle value
        min_throttle = self.throttle_limits[0]  # Minimum throttle value
        self.throttle = min_throttle + (max_throttle - min_throttle) * abs(self.steering)
    