#!/usr/bin/env python3

import cv2
import numpy as np

#local imports
from coordinates_extractor import coordinates_extractor
import constants

# Uses coordinates_extractor to get positions of joints in an image
class joints_locator:

    def __init__(self):
        self.blue_thresholds = np.array([[100, 0, 0], [255, 10, 10]])
        self.green_thresholds = np.array([[0, 100, 0], [10, 255, 10]])
        self.red_thresholds = np.array([[0, 0, 100], [10, 10, 255]])
        self.yellow_thresholds = np.array([[0, 100, 100], [10, 255, 255]])
        self.ce = coordinates_extractor()

        self.y_loc = None
        self.b_loc = None

        self.max_change = 10

        self.prevx = {}
        self.prevy = {}
        self.prevz = {}

    def get_blue_joint(self, img):
        blob = self.ce.get_blob_threshold(img, self.blue_thresholds)
        return self.ce.get_blob_coordinates(blob) +  np.array([0,20])

    def get_green_joint(self, img):
        blob = self.ce.get_blob_threshold(img, self.green_thresholds)
        return self.ce.get_blob_coordinates(blob)

    def get_red_joint(self, img):
        blob = self.ce.get_blob_threshold(img, self.red_thresholds)
        return self.ce.get_blob_coordinates(blob)

    def get_yellow_joint(self, img):
        blob = self.ce.get_blob_threshold(img, self.yellow_thresholds)
        # separate yellow from orange
        bin2 = img.copy()
        bin2[img[:, :, 1] == img[:, :, 2]] = 255
        bin2[img[:, :, 1] != img[:, :, 2]] = 0
        blob = blob & bin2[:, :, 0]
        return self.ce.get_blob_coordinates(blob)

    def get_joints_pixel_location(self, img):
        # Yellow and blue joints are stationary, only need to be found once
        if self.y_loc is None:
            self.y_loc = self.get_yellow_joint(img)
        if self.b_loc is None:
            self.b_loc = self.get_blue_joint(img)
        y = self.y_loc
        b = self.b_loc

        # Green and red could be anywhere
        g = self.get_green_joint(img)
        r = self.get_red_joint(img)
        
        ret = np.array([y, b, g, r])
        print(ret)
        for i in range(4):
            if ret[i] is not None:
                # Origin at the yellow joint
                ret[i,:] -= y
                # Flip y axis (make up positive)
                ret[i,1] *= -1

        return ret

    def combine_2d_imagecoords_into_xyz(self, yz_coords, xz_coords, joint):

        prevx = self.prevx.get(joint)
        prevy = self.prevy.get(joint)
        prevz = self.prevz.get(joint)

        # Determine x
        if xz_coords is None or xz_coords[0] is None:
            x = self.prevx[joint]
        else:
            currx = xz_coords[0]
            if prevx is None:
                diff = 0
            else:
                diff = currx - prevx
            if abs(diff) > self.max_change:
                x = prevx + diff
            else:
                x = currx
            self.prevx[joint] = x
        
        # Determine y
        if yz_coords is None or yz_coords[0] is None:
            y = self.prevy[joint]
        else:
            curry = yz_coords[0]
            if prevy is None:
                diff = 0
            else:
                diff = curry - prevy
            if abs(diff) > self.max_change:
                y = prevy + diff
            else:
                y = curry
            self.prevy[joint] = y
        
        # Determine z
        if (yz_coords is None or yz_coords[1] is None) and (xz_coords is None or xz_coords[1] is None):
            z = self.prevz[joint]
        else:
            if yz_coords is None or yz_coords[1] is None:
                currz = xz_coords[1]
            elif xz_coords is None or xz_coords[1] is None:
                currz = yz_coords[1]
            else:
                currz = (yz_coords[1] + xz_coords[1]) / 2
            if prevz is None:
                diff = 0
            else:
                diff = currz - prevz
            if abs(diff) > self.max_change:
                z = prevz + diff
            else:
                z = currz
            self.prevz[joint] = z

        #print(joint, ,xz_coords, prevx, x, y, z)

        return np.array([x,y,z])


    # Calculates (relative) xyz coordinates of joints, with a frame of reference as shown
    # in figure 1 of the specifications document.
    # img_yz: image from camera1
    # img_xz: image from camera2
    def get_joints_xyz_locations(self, img_yz, img_xz):
        loc1 = self.get_joints_pixel_location(img_yz)
        loc2 = self.get_joints_pixel_location(img_xz)

        ret_coords = np.zeros((4,3))
        joints = ['y','b','g','r']
        for i in range(len(joints)):
            ret_coords[i,:] = self.combine_2d_imagecoords_into_xyz(loc1[i,:], loc2[i,:], joints[i])

        return ret_coords

    # Wrapper for converting coordinates from get_joints_xyz_locations into meters.
    def get_joints_xyz_locations_meters(self, img_yz, img_xz):
        p2m = constants.get_pixels_to_meters_coefficient()
        return self.get_joints_xyz_locations(img_yz, img_xz) * p2m

# test the class
def main():
    jl = joints_locator()
    img = cv2.imread('image_copy.png', cv2.IMREAD_COLOR)
    print(jl.get_joints_pixel_location(img))

if __name__ == '__main__':
    main()
