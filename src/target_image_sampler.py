#!/usr/bin/env python3

import sys
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge, CvBridgeError

#local imports
from target_locator import target_locator
import constants

class target_image_sampler:

    def __init__(self):
        # initialize the node
        rospy.init_node('target_image_sampler', anonymous=True)
        # initialize subscribers to recieve images
        self.image_sub1 = rospy.Subscriber("/camera1/robot/image_raw", Image, self.callback_img1)
        self.image_sub2 = rospy.Subscriber("/camera2/robot/image_raw", Image, self.callback_img2)
        # initialize subscribers to receive actual position of target
        self.target_x_sub = rospy.Subscriber("/target/x_position_controller/command", Float64, self.callback_x)
        self.target_y_sub = rospy.Subscriber("/target/y_position_controller/command", Float64, self.callback_y)
        self.target_z_sub = rospy.Subscriber("/target/z_position_controller/command", Float64, self.callback_z)
        # initialize the bridge between openCV and ROS
        self.bridge = CvBridge()
        # initialize target locator
        self.tl = target_locator()
        # initialize target xyz coordinates and images
        self.target_x = None
        self.target_y = None
        self.target_z = None
        self.img1 = None
        self.img2 = None

        self.is_updated_x = False
        self.is_updated_y = False
        self.is_updated_z = False
        self.is_updated_img1 = False
        self.is_updated_img2 = False
        # initialize samples dimensions
        self.sample_w = 30
        self.sample_h = 30
        # initialize samples count and limit
        self.samples_count = 0
        self.samples_limit = 10000
        # initialize save directory
        self.save_dir = constants.ML_DATA_DIR


    def data_updated(self):
        coords_updated = self.is_updated_x and self.is_updated_y and self.is_updated_z
        images_updated = self.is_updated_img1 and self.is_updated_img2
        return coords_updated and images_updated
    
    def reset_data(self):
        self.is_updated_x = False
        self.is_updated_y = False
        self.is_updated_z = False
        self.is_updated_img1 = False
        self.is_updated_img2 = False

    def take_action_if_updated(self):
        if(self.data_updated()):
            self.reset_data()
            self.take_samples()

    def callback_x(self, data):
        self.target_x = data.data
        self.is_updated_x = True
        self.take_action_if_updated()

    def callback_y(self, data):
        self.target_y = data.data
        self.is_updated_y = True
        self.take_action_if_updated()

    def callback_z(self, data):
        self.target_z = data.data
        self.is_updated_z = True
        self.take_action_if_updated()

    def callback_img1(self, data):
        try:
            conv = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.img1 = conv
            self.is_updated_img1 = True
        except CvBridgeError as e:
            self.is_updated_img1 = False
            print(e)
        self.take_action_if_updated()

    def callback_img2(self, data):
        try:
            conv = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.img2 = conv
            self.is_updated_img2 = True
        except CvBridgeError as e:
            self.is_updated_img2 = False
            print(e)
        self.take_action_if_updated()

    def increase_samples_count(self):
        if self.samples_count + 1 < self.samples_limit:
            self.samples_count += 1
            if self.samples_count % 100 == 0:
                print(self.samples_count)
        else:
            print("DONE!")

    def generate_image_names(self):
        path = self.save_dir
        num = str(self.samples_count)
        self.increase_samples_count()
        sphere_path = path+'sphere_'+num+'.png'
        cuboid_path = path+'cuboid_'+num+'.png'
        return (sphere_path, cuboid_path)

    def take_samples(self):
        if(not self.data_updated):
            return
        img1 = self.img1
        img2 = self.img2
        # take samples from img1
        try:
            sph, cub = self.take_sample_from(img1, self.target_y, self.target_z)
            sph_name , cub_name = self.generate_image_names()
            cv2.imwrite(sph_name, sph)
            cv2.imwrite(cub_name, cub)
        except Exception as e:
            pass
        # take samples from img2
        try:
            sph, cub = self.take_sample_from(img2, self.target_x, self.target_z)
            sph_name , cub_name = self.generate_image_names()
            cv2.imwrite(sph_name, sph)
            cv2.imwrite(cub_name, cub)
        except Exception as e:
            print(e)

        
    def take_sample_from(self, img, x, y):
        blobs = self.tl.get_orange_blobs_img(img)
        blobs_centres = self.tl.get_orange_blobs_centres(img)
        
        if blobs_centres.shape != (2,2):
            # shapes are overlapping, don't take sample
            return None
        xy_actual = np.array([x, y]) * constants.get_meters_to_pixels_coefficient()
        xy_blob1 = blobs_centres[0,:]
        xy_blob2 = blobs_centres[1,:]

        d1 = np.linalg.norm((xy_actual-xy_blob1))
        d2 = np.linalg.norm((xy_actual-xy_blob2))

        if d1 > d2:
            # sphere is blob2
            sph = 1
            cub = 0
        else:
            sph = 0
            cub = 1
        c_sph = blobs_centres[sph,:]
        c_cub = blobs_centres[cub,:]
        centres = [c_sph, c_cub]
        w = self.sample_w
        h = self.sample_h
        ret = np.zeros((2,h,w))
        for i in range(2):
            c = centres[i]
            ul_x = c[0]-w//2
            ul_y = c[1]-h//2
            lr_x = ul_x + w
            lr_y = ul_y + h
            ret[i,:,:] = blobs[ul_y:lr_y, ul_x:lr_x]
        cv2.imshow('Sphere',ret[0,:,:])
        cv2.imshow('Cuboid', ret[1,:,:])
        cv2.waitKey(1)
        return (ret[0,:,:], ret[1,:,:])



# call the class
def main(args):
    tis = target_image_sampler()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutting down")
    cv2.destroyAllWindows()

# run the code if the node is called
if __name__ == '__main__':
    main(sys.argv)
