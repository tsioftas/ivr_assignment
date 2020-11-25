#!/usr/bin/env python3

import cv2
import numpy as np

#local imports
from coordinates_extractor import coordinates_extractor
import constants
from shape_classifier import shape_classifier

# Uses coordinates_extractor to get positions of joints in an image
class target_locator:

    def __init__(self):
        self.orange_thresholds = np.array([[0, 50, 100], [30, 255, 255]])
        self.ce = coordinates_extractor()
        self.sc = shape_classifier()
        self.sample_w = 30
        self.sample_h = 30
        
        self.target_prevx = None
        self.target_prevy = None
        self.target_prevz = None

        # Position of yellow joint in the images. Used to set the origin at the yellow joint
        self.offset = np.array([399, 555])
        # Multiplier used to flip y axis 
        self.multiplier = np.array([1,-1])


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


    def get_sample_from_img(self, img, centre):
        ul_x = centre[0]-self.sample_w//2
        ul_y = centre[1]-self.sample_h//2
        br_x = ul_x + self.sample_w
        br_y = ul_y + self.sample_h
        return img[ul_y:br_y, ul_x:br_x]


    def select_sphere(self, blobs_centres, img):
        blob1 = self.get_sample_from_img(img, blobs_centres[0,:])
        blob2 = self.get_sample_from_img(img, blobs_centres[1,:])

        flat = self.sample_h * self.sample_w
        b1 = np.reshape(blob1, flat)
        b2 = np.reshape(blob2, flat)
        X = np.zeros((2,flat))
        X[0,:] = b1
        X[1,:] = b2
        X = self.sc.normalize(X)

        predictions = self.sc.predict(X)

        if predictions[0] != predictions[1]:
            return int(predictions[0])
        else:
            probabilities = self.sc.p_c_x(X, 0)
            return np.argmax(probabilities)


    def get_target_pixel_location(self, img):
        orange_blobs = self.get_orange_blobs_centres(img)
        if orange_blobs.shape == (2,2):
            sphere = self.select_sphere(orange_blobs, self.get_orange_blobs_img(img))
            return (orange_blobs[sphere, :] - self.offset) * self.multiplier
        elif orange_blobs.shape == (1,2):
            return (orange_blobs[0, :] - self.offset) * self.multiplier
        else:
            return None


    def combine_2d_imagecoords_into_xyz(self, yz_coords, xz_coords):
        # determine x
        if xz_coords is None or xz_coords[0] is None:
            x = self.target_prevx
        else:
            x = xz_coords[0]
            self.target_prevx = x
        # determine y
        if yz_coords is None or yz_coords[0] is None:
            y = self.target_prevy
        else:
            y = yz_coords[0]
            self.target_prevy = y
        # determine z
        if (yz_coords is None or yz_coords[1] is None) and (xz_coords is None or xz_coords[1] is None):
            z = self.target_prevz
        elif yz_coords is None or yz_coords[1] is None:
            z = xz_coords[1]
        elif xz_coords is None or xz_coords[1] is None:
            z = yz_coords[1]
        else:
            z = (yz_coords[1] + xz_coords[1])/2
        self.target_prevz = z

        return np.array([x, y, z])



    # Calculates (relative) xyz coordinates of target, with a frame of reference as shown
    # in figure 1 of the specifications document.
    # img_yz: image from camera1
    # img_xz: image from camera2
    def get_target_xyz_location(self, img_yz, img_xz):
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
