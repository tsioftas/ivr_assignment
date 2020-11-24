#!/usr/bin/env python3

import cv2
import numpy as np

#local imports
from coordinates_extractor import coordinates_extractor
import constants

# Uses coordinates_extractor to get positions of joints in an image
class target_locator:

    def __init__(self):
        self.orange_thresholds = np.array([[0, 50, 100], [30, 255, 255]])
        self.ce = coordinates_extractor()

        self.sphere_loc = None
        self.box_loc = None

    def get_orange_blobs_img(self, img):
        blobs = self.ce.get_blob_threshold(img, self.orange_thresholds)
        # separate yellow from orange
        bin2 = img.copy()
        bin2[img[:, :, 1] == img[:, :, 2]] = 0
        bin2[img[:, :, 1] != img[:, :, 2]] = 255
        blobs = blobs & bin2[:, :, 0]
        return blobs

    def get_orange_blobs_centres(self, img):
        blobs = self.get_orange_blobs_img(img)
        contours = cv2.findContours(blobs, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        ret = []
        for c in contours:
            coords = self.ce.get_blob_coordinates(c)
            if coords is not None:
                ret.append(coords)
        return np.array(ret)

    def select_sphere(blobs):
        # TODO: complete using shape detection
        pass

    def get_target_pixel_location(self, img):
        orange_blobs = self.get_orange_blobs_centres(img)
        sphere = self.select_sphere(orange_blobs)
        return orange_blobs[sphere, :]

    def combine_2d_imagecoords_into_xyz(self, yz_coords, xz_coords):
        # TODO: complete
        pass


    # Calculates (relative) xyz coordinates of target, with a frame of reference as shown
    # in figure 1 of the specifications document.
    # img_yz: image from camera1
    # img_xz: image from camera2
    def get_target_xyz_location(self, img_yz, img_xz):
        # TODO: obstructed vision??
        loc1 = self.get_target_pixel_location(img_yz)
        loc2 = self.get_target_pixel_location(img_xz)

        ret_coords = self.combine_2d_imagecoords_into_xyz(loc1, loc2)

        return ret_coords

    # Wrapper for converting coordinates from get_target_xyz_location into meters.
    def get_target_xyz_location_meters(self, img_yz, img_xz):
        p2m = constants.get_pixels_to_meters_coefficient()
        return self.get_target_xyz_location(img_yz, img_xz) * p2m

# test the class
def main():
    tl = target_locator()
    img = cv2.imread('image_copy.png', cv2.IMREAD_COLOR)
    print(tl.get_target_pixel_location(img))

if __name__ == '__main__':
    main()
