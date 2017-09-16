#!/usr/bin/env python

import rospy
from std_msgs.msg import Bool
from dbw_mkz_msgs.msg import ThrottleCmd, SteeringCmd, BrakeCmd, SteeringReport
from geometry_msgs.msg import TwistStamped, PoseStamped
from styx_msgs.msg import Lane

from speed_controller import SpeedController
from twist_controller import TwistController
from yaw_controller import YawController

from lowpass import LowPassFilter

from dbw_common import get_cross_track_error_from_frenet

class DBWNode(object):

    def __init__(self):
        rospy.init_node('dbw_node')

        # Params:
        vehicle_mass = rospy.get_param('~vehicle_mass', 1736.35)
        fuel_capacity = rospy.get_param('~fuel_capacity', 13.5)
        brake_deadband = rospy.get_param('~brake_deadband', .1)
        decel_limit = rospy.get_param('~decel_limit', -5)
        accel_limit = rospy.get_param('~accel_limit', 1.)
        wheel_radius = rospy.get_param('~wheel_radius', 0.2413)
        wheel_base = rospy.get_param('~wheel_base', 2.8498)
        steer_ratio = rospy.get_param('~steer_ratio', 14.8)
        max_lat_accel = rospy.get_param('~max_lat_accel', 3.)
        max_steer_angle = rospy.get_param('~max_steer_angle', 8.)
        min_speed = 0  # TODO: read from parameter server
        controller_rate = 10  # 30 TODO: read from parameter server, 30 is defined in cpp-file as loop-frequency # 50Hz
        tau_acceleration = 0.5

        self.controller_rate = controller_rate

        # Subscriber:
        self.dbw_enabled = False
        self.current_linear_velocity = None
        self.current_linear_acceleration = None
        self.current_pose = None
        self.target_linear_velocity = None
        self.target_angular_velocity = None

        self.final_waypoints = []
        rospy.Subscriber('/twist_cmd', TwistStamped, self.twist_cmd_cb, queue_size=1)
        rospy.Subscriber('/current_velocity', TwistStamped, self.current_velocity_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.current_pose_cb, queue_size=1)
        rospy.Subscriber('/vehicle/dbw_enabled', Bool, self.dbw_enabled_cb)
        rospy.Subscriber('/final_waypoints', Lane, self.final_waypoints_cb, queue_size=1)

        # Controller:
        self.speed_controller = SpeedController(controller_rate,
                                                accel_limit,
                                                decel_limit,
                                                brake_deadband,
                                                vehicle_mass,
                                                wheel_radius,)

        self.twist_controller = TwistController(controller_rate,
                                                max_steer_angle)

        self.yaw_controller = YawController(wheel_base,
                                            steer_ratio,
                                            min_speed,
                                            max_lat_accel,
                                            max_steer_angle)

        # Filter
        self.lowpass_acceleration = LowPassFilter(tau=tau_acceleration, ts=1.0/self.controller_rate)

        # Publisher:
        self.throttle_pub = rospy.Publisher('/vehicle/throttle_cmd', ThrottleCmd, queue_size=1)
        self.brake_pub = rospy.Publisher('/vehicle/brake_cmd', BrakeCmd, queue_size=1)
        self.steer_pub = rospy.Publisher('/vehicle/steering_cmd', SteeringCmd, queue_size=1)

        # Execute loop for each message (this depends on the defined rate):
        self.loop()

    def twist_cmd_cb(self, msg):
        self.target_linear_velocity = msg.twist.linear.x
        self.target_angular_velocity = msg.twist.angular.z

    def current_velocity_cb(self, msg):
        if self.current_linear_velocity is not None:
            accel = (self.current_linear_velocity - msg.twist.linear.x) * self.controller_rate
            self.current_linear_acceleration = self.lowpass_acceleration.filt(accel)

        self.current_linear_velocity = msg.twist.linear.x

    def final_waypoints_cb(self, msg):
        self.final_waypoints = msg.waypoints

    def current_pose_cb(self, msg):
        self.current_pose = msg.pose

    def dbw_enabled_cb(self, msg):
        if msg.data:
            self.dbw_enabled = True
        else:
            self.dbw_enabled = False
            self.speed_controller.reset()
            self.twist_controller.reset()

    def loop(self):
        rate = rospy.Rate(self.controller_rate)
        while not rospy.is_shutdown():
            rospy.loginfo(self.final_waypoints)
            if self.target_linear_velocity is None \
                    or self.current_linear_velocity is None \
                    or self.current_linear_acceleration is None \
                    or self.current_pose is None \
                    or self.final_waypoints is None:
                continue

            if (len(self.final_waypoints) > 2):

                # Calculate errors
                cross_track_error = get_cross_track_error_from_frenet(self.final_waypoints, self.current_pose)
                rospy.loginfo("cross_track_error is {}".format(cross_track_error))
                steer_twist = self.twist_controller.control(cross_track_error)

                steer_yaw = self.yaw_controller.get_steering(linear_velocity=self.target_linear_velocity,
                                                             angular_velocity=self.target_angular_velocity,
                                                             current_velocity=self.current_linear_velocity)

                throttle, brake = self.speed_controller.control(target_linear_velocity=10,#self.target_linear_velocity,
                                                                current_linear_velocity=self.current_linear_velocity,
                                                                current_linear_acceleration=self.current_linear_acceleration)

                steer = steer_twist + steer_yaw
                #steer = steer_twist
                rospy.loginfo('steer_twist %s', steer_twist)
                rospy.loginfo('steer_yaw %s', steer_yaw)

            else:
                rospy.logwarn('[dbw_node] No more final_waypoints')
                throttle = 0
                brake = 10000
                steer = 0

            if self.dbw_enabled:
                # self.publish(throttle, brake, steer)
                self.publish(throttle, 0, steer)

            rate.sleep() # wiki.ros.org/rospy/Overview/Time#Sleeping_and_Rates --> wait until next rate

    def publish(self, throttle, brake, steer):
        tcmd = ThrottleCmd()
        tcmd.enable = True
        tcmd.pedal_cmd_type = ThrottleCmd.CMD_PERCENT
        tcmd.pedal_cmd = throttle
        self.throttle_pub.publish(tcmd)

        bcmd = BrakeCmd()
        bcmd.enable = True
        bcmd.pedal_cmd_type = BrakeCmd.CMD_TORQUE
        bcmd.pedal_cmd = brake
        self.brake_pub.publish(bcmd)

        scmd = SteeringCmd()
        scmd.enable = True
        scmd.steering_wheel_angle_cmd = steer
        self.steer_pub.publish(scmd)


if __name__ == '__main__':
    DBWNode()
