#!/usr/bin/env python


import rospy

from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint

import scipy.spatial
import math
import numpy as np

LOOKAHEAD_WPS = 30  # 200 requires too much CPU.


def constant_v_waypoints(waypoints, velocity, incremental=True):
    final_waypoints = []

    if incremental:
        forward_velocity = velocity
    else:
        forward_velocity = - velocity


    x = [waypoint.pose.pose.position.x for waypoint in waypoints]
    y = [waypoint.pose.pose.position.y for waypoint in waypoints]
    delta_x = np.diff(x)
    delta_y = np.diff(y)
    delta_s = np.sqrt(np.square(delta_x) + np.square(delta_y))

    last_waypoint = waypoints[0]
    for index, waypoint in enumerate(waypoints[1:]):
        velocity_x = forward_velocity * delta_x[index] / delta_s[index]
        velocity_y = forward_velocity * delta_y[index] / delta_s[index]
        last_waypoint.twist.twist.linear.x = velocity_x
        last_waypoint.twist.twist.linear.y = velocity_y
        final_waypoints.append(last_waypoint)

        last_waypoint = waypoint

    return final_waypoints


def get_closest_waypoint_index(waypoints, pose):

    waypoint_coordinates = [[waypoint.pose.pose.position.x,
                             waypoint.pose.pose.position.y] for waypoint in waypoints]

    pose_coordinates = [pose.pose.position.x, pose.pose.position.y]
    _, index = scipy.spatial.KDTree(waypoint_coordinates).query(pose_coordinates)

    return index


class WaypointUpdater(object):

    def __init__(self):
        rospy.init_node('waypoint_updater')
        # queue_size: after receiving this number of messages, old messages will be deleted
        # queue_size: all py-files seem to have queue_size=1, so I am using 1 here as well
        # exception: the cpp-files in waypoint_follower have queue_size=10
        # todo: test if cpu has less issues with different queue_size and loop-frequency
        self.waypoints = None
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb, queue_size=1)
        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb, queue_size=1)
        self.final_waypoints_pub = rospy.Publisher('/final_waypoints', Lane, queue_size=1)
        rospy.spin()

    def waypoints_cb(self, lane):

        self.waypoints = lane.waypoints

    def pose_cb(self, pose):

        velocity = 20
        if self.waypoints is not None:
            closest_wp_index = get_closest_waypoint_index(self.waypoints, pose)
            waypoints_2laps = self.waypoints + self.waypoints
            lane = Lane()
            lane.waypoints = constant_v_waypoints(waypoints_2laps[closest_wp_index:
                                                                  closest_wp_index+LOOKAHEAD_WPS],
                                                  velocity)
            self.final_waypoints_pub.publish(lane)


    '''
    def traffic_cb(self, msg):
        # TODO: Callback for /traffic_waypoint message. Implement
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist
    '''

if __name__ == '__main__':
    try:
        WaypointUpdater()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
