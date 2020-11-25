#!/usr/bin/env python3

import roslib
import sys
import rospy
import numpy as np
from std_msgs.msg import Float64
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# local imports
from target_locator import target_locator

class target_tracker:

    def __init__(self):
        # initialize the node
        rospy.init_node('track_target_2.2.py', anonymous=True)
        self.rate = rospy.Rate(30) # 30hz
        # initialize subscribers to receive images
        self.image_sub1 = rospy.Subscriber("image_topic1", Image, self.record_img1)
        self.image_sub2 = rospy.Subscriber("image_topic2", Image, self.record_img2)
        # initialize publishers to publish estimated coordinates
        self.targ_x_pub = rospy.Publisher("target_x_estimate",Float64, queue_size=10)
        self.targ_y_pub = rospy.Publisher("target_y_estimate",Float64, queue_size=10)
        self.targ_z_pub = rospy.Publisher("target_z_estimate",Float64, queue_size=10)
        # initialize images
        self.img1 = None
        self.img2 = None
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize target locator
        self.tl = target_locator()

    
    def record_img1(self, data):
        try:
            self.img1 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def record_img2(self, data):
        try:
            self.img2 = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)


    def track(self):
        t0 = rospy.get_time()
        t = 0
        # run for 10.5 seconds
        while t < 10.5:
            t = (np.array([rospy.get_time()]) - t0)
            print(t)
            if (self.img1 is not None) and (self.img2 is not None):
                coords = self.tl.get_target_xyz_location_meters(self.img1, self.img2)
                x = Float64()
                x.data = coords[0]
                y = Float64()
                y.data = coords[1]
                z = Float64()
                z.data = coords[2]

                self.targ_x_pub.publish(x)
                self.targ_y_pub.publish(y)
                self.targ_z_pub.publish(z)
            self.rate.sleep()


# run the code if the node is called
if __name__ == '__main__':
  try:
    tt = target_tracker()
    tt.track()
  except rospy.ROSInterruptException:
    pass


