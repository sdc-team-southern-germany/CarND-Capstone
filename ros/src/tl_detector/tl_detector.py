#!/usr/bin/env python
import rospy
from std_msgs.msg import Int32
from geometry_msgs.msg import PoseStamped, Pose
from styx_msgs.msg import TrafficLightArray, TrafficLight
from styx_msgs.msg import Lane
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from light_classification.tl_classifier import TLClassifier
import tf
#import cv2
import yaml
import os
import numpy as np
from scipy import spatial
from scipy.misc import imshow
from scipy.misc import imsave

STATE_COUNT_THRESHOLD = 3
GET_TRAINING_DATA = False        # Set to True if you want to save training data
SIM_DATA_PATH = os.getcwd()+ "/light_classification/data/simulator/"

class TLDetector(object):
    def __init__(self):
        rospy.init_node('tl_detector')

        self.pose = None
        self.waypoints = None
        self.camera_image = None
        self.lights = []
        self.state = None


        self.cust_waypoints = []
        self.cust_tlights = []
        self.detected_tlight = None
        self.dist = None

        self.bridge = CvBridge()
        self.light_classifier = TLClassifier()
        self.listener = tf.TransformListener()
        self.state = TrafficLight.UNKNOWN
        self.last_state = TrafficLight.UNKNOWN
        self.last_wp = -1
        self.state_count = 0

        sub1 = rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        sub2 = rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        '''
        /vehicle/traffic_lights provides you with the location of the traffic light in 3D map space and
        helps you acquire an accurate ground truth data source for the traffic light
        classifier by sending the current color state of all traffic lights in the
        simulator. When testing on the vehicle, the color state will not be available. You'll need to
        rely on the position of the light and the camera image to predict it.
        '''
        config_string = rospy.get_param("/traffic_light_config")
        self.config = yaml.load(config_string)

        if GET_TRAINING_DATA:
            if os.path.exists(SIM_DATA_PATH+"label.txt"):
                labels_path = os.path.join(SIM_DATA_PATH, 'label.txt')
                label_no = np.loadtxt(labels_path, dtype=int, delimiter=' ', skiprows=1, usecols=(0,))
                self.counter = label_no[-1] + 1
            else:
                self.counter = 1
            #if os.path.exists(SIM_DATA_PATH+"label.txt"):
                #os.remove(SIM_DATA_PATH+"label.txt")

        self.upcoming_red_light_pub = rospy.Publisher('/traffic_waypoint', Int32, queue_size=1)
        sub3 = rospy.Subscriber('/vehicle/traffic_lights', TrafficLightArray, self.traffic_cb)
        sub6 = rospy.Subscriber('/image_color', Image, self.image_cb)

        rospy.spin()

    def pose_cb(self, msg):
        #print("[TLD] pose attr: ", dir(self.pose))
        #print("pose: ", self.pose.position.x, self.pose.position.y)
        self.pose = msg


    def waypoints_cb(self, waypoints):
        # print("[TLD]waypoint dir: \n", dir(waypoints)) # print all attribute of the class
        if not self.waypoints:
            self.waypoints = waypoints
            for i in range(0, len(self.waypoints.waypoints)):
                self.cust_waypoints.append([self.waypoints.waypoints[i].pose.pose.position.x, self.waypoints.waypoints[i].pose.pose.position.y])


    def traffic_cb(self, msg):
        #print("[TLD] TL: ", dir(self.lights))
        #print("light[0] ", self.lights[0])
        if not self.lights:
            self.lights = msg.lights

            for i in range(0, len(self.lights)):
                tl_pose=[self.lights[i].pose.pose.position.x,self.lights[i].pose.pose.position.y]
                self.cust_tlights.append(tl_pose)

    def image_cb(self, msg):
        """Identifies red lights in the incoming camera image and publishes the index
            of the waypoint closest to the red light's stop line to /traffic_waypoint

        Args:
            msg (Image): image from car-mounted camera

        """
        self.has_image = True
        self.camera_image = msg
        light_wp, state = self.process_traffic_lights()

        '''
        Publish upcoming red lights at camera frequency.
        Each predicted state has to occur `STATE_COUNT_THRESHOLD` number
        of times till we start using it. Otherwise the previous stable state is
        used.
        '''
        if self.state != state:
            self.state_count = 0
            self.state = state
            print("TL state: ", state)
        elif self.state_count >= STATE_COUNT_THRESHOLD:
            self.last_state = self.state
            light_wp = light_wp if state == TrafficLight.RED else -1
            #print("RED light wp_index: ", light_wp)
            self.last_wp = light_wp
            self.upcoming_red_light_pub.publish(Int32(light_wp))
        else:
            self.upcoming_red_light_pub.publish(Int32(self.last_wp))
        self.state_count += 1

    def get_closest_waypoint(self, pose):
        """Identifies the closest path waypoint to the given position
            https://en.wikipedia.org/wiki/Closest_pair_of_points_problem
        Args:
            pose (Pose): position to match a waypoint to

        Returns:
            int: index of the closest waypoint in self.waypoints

        """
        # YWiyogo: this is a nearest neighbours search problem, not a closest pair of points problem
        # Sort waypoint

        # Divide the x_n points and compare the distance to the x_(n/2) & x_(n/2)+1
        if self.waypoints and pose:
            # tree = KDTree(X, leaf_size=2)
            cust_pose = [pose.position.x, pose.position.y]
            # dist, ind = tree.query(cust_pose, k=1)
            dist,ind = spatial.KDTree(self.cust_waypoints).query(cust_pose)
            #print("[TLD]pose: \n", pose)
            #if(ind > len(self.waypoints.waypoints) or ind <0):
                #print("[TDL]Err index out of range %d" % ind)
            #else:
                #print("[TLD] Closest index %d, distance to wp: %f, x: %f, y: %f" % (
                #ind, dist, self.waypoints.waypoints[ind].pose.pose.position.x, self.waypoints.waypoints[ind].pose.pose.position.y))
            return ind
        else:
            return -1


    def project_to_image_plane(self, point_in_world):
        """Project point from 3D world coordinates to 2D camera image location

        Args:
            point_in_world (Point): 3D location of a point in the world

        Returns:
            x (int): x coordinate of target point in image
            y (int): y coordinate of target point in image

        """

        fx = self.config['camera_info']['focal_length_x']
        fy = self.config['camera_info']['focal_length_y']
        image_width = self.config['camera_info']['image_width']
        image_height = self.config['camera_info']['image_height']

        # get transform between pose of camera and world frame
        trans = None
        try:
            now = rospy.Time.now()
            self.listener.waitForTransform("/base_link",
                  "/world", now, rospy.Duration(1.0))
            (trans, rot) = self.listener.lookupTransform("/base_link",
                  "/world", now)

        except (tf.Exception, tf.LookupException, tf.ConnectivityException):
            rospy.logerr("Failed to find camera to map transform")

        #YW Use tranform and rotation to calculate 2D position of light in image

        trans_matrix = self.listener.fromTranslationRotation(trans, rot)
        pt_world_vec = np.array([[point_in_world.x],
                                 [point_in_world.y],
                                 [point_in_world.z],
                                 [1.0]])
        cam_vec = np.dot(trans_matrix, pt_world_vec)

        pt_in_cam = Point(cam_vec[0][0], cam_vec[1][0], cam_vec[2][0])

        # Note: X points forward direction, Y to the left directionn and Z up
        # See image projection formula
        x = int( -1 * (fx / pt_in_cam.x) * pt_in_cam.y)
        y = int( -1 * (fy / pt_in_cam.x) * pt_in_cam.z)

        return (x, y)

    def get_light_state(self, light, gt_state):
        """Determines the current color of the traffic light
        Args:
            light (TrafficLight): light to classify
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """
        if(not self.has_image):
            self.prev_light_loc = None
            return False

        cv_image = self.bridge.imgmsg_to_cv2(self.camera_image, "bgr8")

        # Display camera image
        enable_imshow = False    #activate to see the camera image
        if enable_imshow:
            #cv2.imshow("Image window", cv_image)
            #cv2.waitKey(1)
            scipy.misc.imshow(images[0])

        # Commented out for testing...
        #x, y = self.project_to_image_plane(light.pose.pose.position)

        #TODO use light location to zoom in on traffic light in image

        #Get classification
        state = self.light_classifier.get_classification(cv_image)


        # Get training data set
        if GET_TRAINING_DATA:
            self.get_training_data(cv_image, gt_state)

        return state

    def process_traffic_lights(self):
        """Finds closest visible traffic light, if one exists, and determines its
            location and color
        Returns:
            int: index of waypoint closest to the upcoming stop line for a traffic light (-1 if none exists)
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """

        light = None

        # List of positions that correspond to the line to stop in front of for a given intersection
        stop_line_positions = self.config['stop_line_positions']
        # NOTE, YW: the above stop_line_positions is based on the sim_traffic_light_config.xml
        # but the entries are not the same as from the traffic_cb function !!
        # Currently I concern only the traffic_cb
        #print("light pos: ", stop_line_positions)
        if self.pose and len(self.cust_tlights) > 0:
            #YW: find the closest visible traffic light (if one exists)
            cust_pose = [self.pose.pose.position.x, self.pose.pose.position.y]
            dist, ind = spatial.KDTree(self.cust_tlights).query(cust_pose)
            if (self.dist == None):
                self.dist = dist

            #diff_x = self.cust_tlights[ind][0] - self.pose.pose.position.x
            diff_dist = self.dist - dist
            self.dist = dist
            cam_dist_to_tl = 250    # in m
            if(dist < cam_dist_to_tl):
                if diff_dist >= 0:
                    # get the index of the closest waypoint from the TL pose
                    light = self.get_closest_waypoint(self.lights[ind].pose.pose)

                    if self.detected_tlight != self.cust_tlights[ind]:
                        self.detected_tlight = self.cust_tlights[ind]
                        #print("[TLD] TL %d found, A front distance to current pose: %f, waypoint index: %d" % (ind, dist, light))
                        print("[TLD] TL %d found %dm ahead of current pose at waypoint index %d" % (ind, dist, light))
            else:
                self.detected_tlight = None
                self.state = TrafficLight.UNKNOWN
                self.state_count = 0

            gt_state = self.lights[ind].state
            if light:
                state = self.get_light_state(light, gt_state)
                return light, state

        return -1, TrafficLight.UNKNOWN


    def get_training_data(self, image, label):
        """Gets training data set from the simulator
        """

        filename= SIM_DATA_PATH + "TL_"+str(self.counter)+".png"
        #cv2.imwrite(filename,image)
        imsave(filename,image)
        if os.path.exists(SIM_DATA_PATH+"label.txt"):
            append_write = 'a' # append if already exists
        else:
            append_write = 'w' # make a new file if not

        f = open(SIM_DATA_PATH+"label.txt", append_write)
        str_line = "%d %.0f %.0f %d\n" % (self.counter, self.pose.pose.position.x, self.pose.pose.position.y, label)
        f.write(str_line)  # python will convert \n to os.linesep
        f.close()
        self.counter = self.counter+1


if __name__ == '__main__':
    try:
        TLDetector()
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start traffic node.')
