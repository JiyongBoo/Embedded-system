import datetime # don't forget to import datetime

class SteeringController:
    def __init__(self, Kp, Ki, Kd, setpoint=0, steering_limits=(-1.0, 1.0)):
 
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.integral = 0.0         # cumulative error
        self.pre_error = 0.0        # previous error
        self.pre_t = None           # previous time
        self.setpoint = setpoint    # Your goal
        self.steering_limits = steering_limits  # (min_output, max_output)
        self.steering = 0.0
        self.Ka = 2.0/Kp        # Anti-Windup coefficient
        self.anti = 0.0
    
    def update(self, output):
        # output must be "current value from sensor", not the actual output of the function
        # The actual output of the function is "self.steering" & "self.throttle". You should call this variable to use for car.steering & car.throttle
        if output is None:
            self.pre_t = None
            return
        error = self.setpoint - output 

        cur_t = datetime.datetime.now()  # needs to import `datetime`
        if self.pre_t is None:
            dt = 0.0
        else:
            dt = (cur_t - self.pre_t).total_seconds()

        self.pre_t = cur_t

        if dt > 0.0:
            # self.integral += error * dt   # without anti-windup
            self.integral += (error - self.Ka * self.anti) * dt
            derivative = (error - self.pre_error) / dt
        else:
            # when dt == 0, prevent dividing zero
            derivative = 0.0

        self.pre_error = error

        self.steering = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        
        if self.steering > self.steering_limits[1]:
            saturated = self.steering_limits[1]
        elif self.steering < self.steering_limits[0]:
            saturated = self.steering_limits[0]
        else:
            saturated = self.steering
        self.anti = self.steering - saturated
        
        self.steering = saturated

class ThrottleController:
    def __init__(self, tolerances_a:tuple, tolerances_b:tuple, throttle_limits=(0, 0.5), throttle_gain=0.02, throttle_decrease_rate=0.03):
        self.tolerances_a: tuple = tolerances_a
        self.tolerances_b: tuple = tolerances_b
        self.throttle_limits = throttle_limits
        self.pre_t = None
        self.previous_features = (0,0)
        self.throttle = 0.0
        self.throttle_reference = 0.0
        self.throttle_gain = throttle_gain
        self.throttle_decrease_rate = throttle_decrease_rate

    
    def update(self, throttle_reference:float, features:tuple) -> None:

        differences = tuple(abs(f - pf) for (f, pf) in zip(features, self.previous_features))

        cur_t = datetime.datetime.now()
        if self.pre_t is None:
            dt = 0.0
        else:
            dt = (cur_t - self.pre_t).total_seconds()

        if (throttle_reference != self.throttle_reference): # when new throttle_reference is set
            self.throttle = throttle_reference
            self.throttle_reference = throttle_reference
        elif (throttle_reference == 0):
            # make car stop forever
            # do not increase throttle
            self.throttle = 0.0
            self.throttle_reference = 0.0
        elif all((diff < tol) for (diff, tol) in zip(differences, self.tolerances_a)):  # or you can use `any`, tolerances_a < tolerances_b
            self.throttle += self.throttle_gain * dt * (1 if throttle_reference > 0 else -1)
        elif all((diff < tol) for (diff, tol) in zip(differences, self.tolerances_b)):  # or you can use `any`
            pass
        elif (abs(self.throttle) > abs(throttle_reference)):
            # FIXME: What happens if throttle is larger than 0 and throttle_reference is negative?
            self.throttle -= self.throttle_decrease_rate * self.throttle_gain * dt * (1 if throttle_reference > 0 else -1)

        if (self.throttle > self.throttle_limits[1]):
            self.throttle = self.throttle_limits[1]
        elif (self.throttle < self.throttle_limits[0]):
            self.throttle = self.throttle_limits[0]
        
        self.previous_features = features
        self.pre_t = cur_t