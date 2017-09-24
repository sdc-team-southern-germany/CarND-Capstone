import pid
import lowpass
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class TwistController(object):
    def __init__(self, controller_rate, max_steer_angle):

        self.controller_rate = controller_rate

        # TODO: find good parameters for PID controllers
        self.pid_steer = pid.PID(kp=0.1, ki=0.00, kd=0.25,
                                 mn=-max_steer_angle, mx=max_steer_angle)

        self.last_timestamp = rospy.get_time()

    def control(self, cross_track_error):
        """
        calculates control signal
        :param speed_error:
        :param cross_track_error: The cross track error (cte) is the current y
                position of the vehicle
        :return: throttle [1], brake [Nm], steer [rad]
        """
        current_timestamp = rospy.get_time()
        duration = current_timestamp - self.last_timestamp
        self.last_timestamp = current_timestamp

        #rospy.loginfo('Error cte: %s', cross_track_error)
        steering_angle = self.pid_steer.step(error=cross_track_error,
                                             sample_time=duration)

        steering_out = steering_angle

        return steering_out

    def reset(self):
        """
        resets pid controllers
        """

        self.pid_steer.reset()

def test_controller():
    """
    simple test of functionality by creating Controller object and performing
    one control step
    """
    controller = TwistController(controller_rate=30,
                                 max_steer_angle=0.25)

    controller.control(0.1)

if __name__ == "__main__":
    test_controller()
