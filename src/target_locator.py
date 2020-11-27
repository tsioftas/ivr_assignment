#!/usr/bin/env python3

import cv2
import numpy as np

#local imports
from coordinates_extractor import coordinates_extractor
import constants
from shape_classifier import shape_classifier

class target_locator:
    """Class used to get the xyz position of the target (orange sphere), given
    the views from the two cameras"""

    def __init__(self):
        # Threshold to extract orange colored blobs
        self.orange_thresholds = np.array([[0, 50, 100], [30, 255, 255]])
        # Initialise coordinates_extractor
        self.ce = coordinates_extractor()
        # Initialise shape_classifier
        self.sc = shape_classifier()
        # Initialise sample dimensions for shape_classifier
        self.sample_w = 30
        self.sample_h = 30
        # Initialise last seen xyz coordinates of target
        self.prevx = [None, None] # [target, cuboid]
        self.prevy = [None, None]
        self.prevz = [None, None]
        # Position of yellow joint in the images. Used to set the origin at the yellow joint
        self.offset = constants.YELLOW_PIXEL_LOCATION
        # Multiplier used to flip y axis
        self.multiplier = np.array([1, -1])


    # Given an image returns a binary image showing the orange-coloured regions
    def get_orange_blobs_img(self, img):
        blobs = self.ce.get_blob_threshold(img, self.orange_thresholds)
        # separate yellow from orange
        bin2 = img.copy()
        bin2[img[:, :, 1] == img[:, :, 2]] = 0
        bin2[img[:, :, 1] != img[:, :, 2]] = 255
        blobs = blobs & bin2[:, :, 0]
        return blobs


    # Given an image extract the centres of the visible orange blobs. If no orange blobs
    # are visible, an empty array is returned
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


    # Given an image and a centre returns a sample
    # taken from the image centered at the center
    def get_sample_from_img(self, img, centre):
        ul_x = centre[0]-self.sample_w//2
        ul_y = centre[1]-self.sample_h//2
        br_x = ul_x + self.sample_w
        br_y = ul_y + self.sample_h
        return img[ul_y:br_y, ul_x:br_x]


    # Given an image and two blobs centres, returns the index of the blob
    # that is a sphere (or is closer to a sphere). This is done using the
    # shape_classifier.
    def select_sphere(self, blobs_centres, img):
        # Get sample images for each blob
        blob1 = self.get_sample_from_img(img, blobs_centres[0, :])
        blob2 = self.get_sample_from_img(img, blobs_centres[1, :])
        # Flatten the images and normalize
        flat = self.sample_h * self.sample_w
        b1 = np.reshape(blob1, flat)
        b2 = np.reshape(blob2, flat)
        X = np.zeros((2, flat))
        X[0, :] = b1
        X[1, :] = b2
        X = self.sc.normalize(X)
        # Get predicted labels for each blob
        predictions = self.sc.predict(X)
        # If prefictions differ, return the one the index
        # whose prediction is 0 (sphere). Return the index of
        # the blob most likely to be a sphere according to the model.
        if predictions[0] != predictions[1]:
            return int(predictions[0])
        else:
            probabilities = self.sc.p_c_x(X, 0)
            return np.argmax(probabilities)


    # Given an image returns the pixel coordinates of the orange sphere, centered at
    # the yellow joint, oriented as the axes in figure1 of the specifications document indicate.
    def get_target_pixel_location(self, img, both=False):
        orange_blobs = self.get_orange_blobs_centres(img)
        if orange_blobs.shape == (2, 2):
            # Two blobs, use shape_classifier to determine which one is the shpere
            sphere = self.select_sphere(orange_blobs, self.get_orange_blobs_img(img))
            if not both:
                return (orange_blobs[sphere, :] - self.offset) * self.multiplier
            else:
                return [(orange_blobs[sphere, :] - self.offset) * self.multiplier, (orange_blobs[1-sphere, :] - self.offset) * self.multiplier]
        elif orange_blobs.shape == (1, 2):
            # One blob, probably the objects are overlapping. Assume they have the same coordinates
            if not both:
                return (orange_blobs[0, :] - self.offset) * self.multiplier
            else:
                return [(orange_blobs[0, :] - self.offset) * self.multiplier, (orange_blobs[0, :] - self.offset) * self.multiplier]
        else:
            if not both:
                return None
            else:
                return [None, None]


    # Given the pixel coordinates of the target in the two camera views,
    # returns its xyz pixel coordinates. If one of the coordinates is not
    # available, the last seen value for that coordinate is used.
    def combine_2d_imagecoords_into_xyz(self, yz_coords, xz_coords, which=0):
        # determine x
        if xz_coords is None or xz_coords[0] is None:
            x = self.prevx[which]
        else:
            x = xz_coords[0]
            self.prevx[which] = x
        # determine y
        if yz_coords is None or yz_coords[0] is None:
            y = self.prevy[which]
        else:
            y = yz_coords[0]
            self.prevy[which] = y
        # determine z
        if (yz_coords is None or yz_coords[1] is None) and \
            (xz_coords is None or xz_coords[1] is None):
            z = self.prevz[which]
        elif yz_coords is None or yz_coords[1] is None:
            z = xz_coords[1]
        elif xz_coords is None or xz_coords[1] is None:
            z = yz_coords[1]
        else:
            z = (yz_coords[1] + xz_coords[1])/2
        self.prevz[which] = z

        return np.array([x, y, z])



    # Calculates (relative) xyz coordinates of target, with a frame of reference as shown
    # in figure 1 of the specifications document.
    # img_yz: image from camera1
    # img_xz: image from camera2
    def get_target_xyz_location(self, img_yz, img_xz, both=False):
        loc1 = self.get_target_pixel_location(img_yz, both=both)
        loc2 = self.get_target_pixel_location(img_xz, both=both)

        if not both:
            ret_coords = self.combine_2d_imagecoords_into_xyz(loc1, loc2)
        else:
            ret_coords = np.zeros((2,3))
            ret_coords[0,:] = (self.combine_2d_imagecoords_into_xyz(loc1[0], loc2[0], 0))
            ret_coords[1,:] = (self.combine_2d_imagecoords_into_xyz(loc1[1], loc1[1], 1))

        return ret_coords

    # Wrapper for converting coordinates from get_target_xyz_location into meters.
    def get_target_xyz_location_meters(self, img_yz, img_xz, both=False):
        p2m = constants.get_pixels_to_meters_coefficient()
        return self.get_target_xyz_location(img_yz, img_xz, both=both) * p2m
    
