#!/usr/bin/env python

import math
import numpy as np
import rospy


def distance(x1, y1, x2, y2):
    return float(np.sqrt((x2-x1)**2 + (y2-y1)**2))

def get_cross_track_error_from_frenet(waypoints, pose):

    n_x = waypoints[1].pose.pose.position.x - waypoints[0].pose.pose.position.x
    n_y = waypoints[1].pose.pose.position.y - waypoints[0].pose.pose.position.y
    x_x = pose.position.x - waypoints[0].pose.pose.position.x
    x_y = pose.position.y - waypoints[0].pose.pose.position.y

    assert ((n_x * n_x + n_y * n_y) != 0);

    # find the projection of x onto n
    proj_norm = (x_x * n_x + x_y * n_y) / (n_x * n_x + n_y * n_y)
    proj_x = proj_norm * n_x
    proj_y = proj_norm * n_y

    frenet_d = distance(x_x, x_y, proj_x, proj_y)

    sign = np.sign(np.cross([x_x, x_y, 0], [n_x, n_y, 0])[-1])

    if (sign < 0):
        rospy.logwarn('Changed sign in cte calculation')
        frenet_d *= -1;

    return frenet_d


def test_get_cross_track_error():

    pass


if __name__ == "__main__":
    test_get_cross_track_error()
