import pid
import lowpass
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class SpeedController(object):
    def __init__(self, controller_rate, accel_limit, decel_limit, brake_deadband, vehicle_mass, wheel_radius):

        self.controller_rate = controller_rate
        self.accel_limit = accel_limit
        self.decel_limit = decel_limit
        self.brake_deadband = brake_deadband
        self.vehicle_mass = vehicle_mass
        self.wheel_radius = wheel_radius

        self.pid_acceleration = pid.PID(kp=0.2, ki=0.05, kd=0.05, mn=self.decel_limit, mx=self.accel_limit)
        self.pid_throttle = pid.PID(kp=0.3, ki=0.01, kd=0.025, mn=0.0, mx=0.4)

        self.last_timestamp = rospy.get_time()


    def control(self, target_linear_velocity, current_linear_velocity, current_linear_acceleration):
        """
        calculates control signal
        :param speed_error:
        :param cross_track_error: The cross track error (cte) is the current y
                position of the vehicle
        :return: throttle [1], brake [Nm], steer [rad]
        """

        velocity_error = target_linear_velocity - current_linear_velocity

        rospy.loginfo("velocity_error is {}".format(velocity_error))

        current_timestamp = rospy.get_time()
        duration = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp

        sample_time = 1.0 / self.controller_rate

        rospy.loginfo('Error speed: %s', velocity_error)

        acceleration_cmd = self.pid_acceleration.step(error=velocity_error, sample_time=duration)
        rospy.loginfo('Acc PID output: %s', acceleration_cmd)

        acceleration_error = acceleration_cmd - current_linear_acceleration

        if (acceleration_cmd < self.brake_deadband):
            throttle_out = 0
            brake_out = -acceleration_cmd * self.vehicle_mass * self.wheel_radius / 4
            self.pid_acceleration.reset()
            rospy.logdebug('[speed_controller] Really braking')
        elif (acceleration_cmd < 0):
            throttle_out = 0
            brake_out = 0
            self.pid_acceleration.reset()
            rospy.logdebug('[speed_controller] Using brake deadband')
        else:
            throttle_out = self.pid_throttle.step(error=acceleration_error, sample_time=sample_time)
            rospy.loginfo('Throttle PID output %s', throttle_out)
            brake_out = 0

        if (current_linear_velocity == 0 and throttle_out == 0):
            self.pid_throttle.reset()


        return throttle_out, brake_out

    def reset(self):
        """
        resets pid controllers
        """
        self.pid_throttle.reset()


def test_controller():
    """
    simple test of functionality by creating Controller object and performing
    one control step
    """
    controller = SpeedController(controller_rate=30,
                            accel_limit=10,
                            decel_limit=-10,
                            vehicle_mass=1000,
                            wheel_radius=0.3)

    controller.control(1)

if __name__ == "__main__":
    test_controller()
