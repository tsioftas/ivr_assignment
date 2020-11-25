#!/usr/bin/env python3

import numpy as np

#local imports
from coordinates_extractor import coordinates_extractor
import constants


class joints_locator:
    """Contains methods to extract the xyz positions of the joints
    when given images from camera1 and camera2"""

    def __init__(self):
        # Threshold values for different colors
        self.blue_thresholds = np.array([[100, 0, 0], [255, 10, 10]])
        self.green_thresholds = np.array([[0, 100, 0], [10, 255, 10]])
        self.red_thresholds = np.array([[0, 0, 100], [10, 10, 255]])
        self.yellow_thresholds = np.array([[0, 100, 100], [10, 255, 255]])
        # Initialize coordinates extractor
        self.ce = coordinates_extractor()
        # Initialize previous coordinates dictionaries. These will contain
        # the last seen x, y and z pixel coordinates for each joint. To be used
        # when a joint is not visible.
        self.prevx = {}
        self.prevy = {}
        self.prevz = {}

    # Methods for getting the pixel coordinates of a joint in an image.
    # Will return None if the joint is not visible
    def get_green_joint(self, img):
        blob = self.ce.get_blob_threshold(img, self.green_thresholds)
        return self.ce.get_blob_coordinates(blob)

    def get_red_joint(self, img):
        blob = self.ce.get_blob_threshold(img, self.red_thresholds)
        return self.ce.get_blob_coordinates(blob)


    # Method that given an image returns the pixel coordinates
    # of joints in that image, centered at the yellow joint and with the y-axis pointing up
    def get_joints_pixel_location(self, img):
        # Yellow and blue joints are stationary
        y = constants.YELLOW_PIXEL_LOCATION
        b = constants.BLUE_PIXEL_LOCATION
        # Green and red could be anywhere
        g = self.get_green_joint(img)
        r = self.get_red_joint(img)
        ret = np.array([y, b, g, r])
        for i in range(4):
            if ret[i] is not None:
                # Origin at the yellow joint
                ret[i, :] -= y
                # Flip y axis (make up positive)
                ret[i, 1] *= -1
        return ret


    # Method that combines 2d pixel coordinates from the two camera views into 3d pixel
    # coordinates, for a specific joint. If one of the coordinates is not available, the
    # last seen coordinate is used.
    #
    # joint: character 'y', 'b', 'g' or 'r' indicating what joint is being dealt with
    def combine_2d_imagecoords_into_xyz(self, yz_coords, xz_coords, joint):
        # Determine x
        if xz_coords is None or xz_coords[0] is None:
            x = self.prevx[joint]
        else:
            x = xz_coords[0]
            self.prevx[joint] = x
        # Determine y
        if yz_coords is None or yz_coords[0] is None:
            y = self.prevy[joint]
        else:
            y = yz_coords[0]
            self.prevy[joint] = y
        # Determine z
        if (yz_coords is None or yz_coords[1] is None) and \
            (xz_coords is None or xz_coords[1] is None):
            z = self.prevz[joint]
        else:
            if yz_coords is None or yz_coords[1] is None:
                z = xz_coords[1]
            elif xz_coords is None or xz_coords[1] is None:
                z = yz_coords[1]
            else:
                z = (yz_coords[1] + xz_coords[1]) / 2
            self.prevz[joint] = z
        return np.array([x, y, z])


    # Given two camera views alculates xyz pixel coordinates of joints,
    # with a frame of reference as shown in figure 1 of the specifications document.
    # img_yz: image from camera1
    # img_xz: image from camera2
    def get_joints_xyz_locations(self, img_yz, img_xz):
        # Get pixel locations for each image
        loc1 = self.get_joints_pixel_location(img_yz)
        loc2 = self.get_joints_pixel_location(img_xz)
        # Combine to get 3d coordinates
        ret_coords = np.zeros((4, 3))
        joints = ['y', 'b', 'g', 'r']
        for i in range(4):
            ret_coords[i, :] = self.combine_2d_imagecoords_into_xyz(loc1[i, :],
                                                                    loc2[i, :],
                                                                    joints[i])
        return ret_coords


    # Wrapper for converting coordinates from get_joints_xyz_locations into meters.
    def get_joints_xyz_locations_meters(self, img_yz, img_xz):
        p2m = constants.get_pixels_to_meters_coefficient()
        return self.get_joints_xyz_locations(img_yz, img_xz) * p2m
