#!/usr/bin/env python3

import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# local imports
from target_locator import target_locator

class target_tracker:
    """Used to track the xyz position of the target using the view from the two cameras
    and publish its estimated coordinates"""

    def __init__(self):
        # initialize the node
        rospy.init_node('target_tracker', anonymous=True)
        # initialize publishers to publish estimated coordinates
        self.targ_x_pub = rospy.Publisher("target_x_estimate", Float64, queue_size=10)
        self.targ_y_pub = rospy.Publisher("target_y_estimate", Float64, queue_size=10)
        self.targ_z_pub = rospy.Publisher("target_z_estimate", Float64, queue_size=10)
        self.obs_x_pub = rospy.Publisher("obstacle_x_estimate", Float64, queue_size=10)
        self.obs_y_pub = rospy.Publisher("obstacle_y_estimate", Float64, queue_size=10)
        self.obs_z_pub = rospy.Publisher("obstacle_z_estimate", Float64, queue_size=10)
        # initialize images
        self.img1 = None
        self.img2 = None
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize target locator
        self.tl = target_locator()
        # initialize subscribers to receive images
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.record_img1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.record_img2)
        self.is_tracking = False

    # Callback functions for saving the images
    def record_img1(self, data):
        try:
            self.img1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)
        if not self.is_tracking:
            self.is_tracking = True
            self.track()

    def record_img2(self, data):
        try:
            self.img2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    # Track the position of the target.
    def track(self):
        if (self.img1 is not None) and (self.img2 is not None):
            coords_both = self.tl.get_target_xyz_location_meters(self.img1, self.img2, both=True)
            print(coords_both)
            x = Float64()
            x.data = coords_both[0, 0]
            y = Float64()
            y.data = coords_both[0, 1]
            z = Float64()
            z.data = coords_both[0, 2]
            self.targ_x_pub.publish(x)
            self.targ_y_pub.publish(y)
            self.targ_z_pub.publish(z)
            x = Float64()
            x.data = coords_both[1, 0]
            y = Float64()
            y.data = coords_both[1, 1]
            z = Float64()
            z.data = coords_both[1, 2]
            self.obs_x_pub.publish(x)
            self.obs_y_pub.publish(y)
            self.obs_z_pub.publish(z)
        self.is_tracking = False


# run the code if the node is called
if __name__ == '__main__':
    try:
        tt = target_tracker()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
