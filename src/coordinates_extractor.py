#!/usr/bin/env python3

import cv2
import numpy as np

# local imports
import math_utils as mu

# Class used to extract object coordinates from images using color thresholding
class coordinates_extractor:

    def __init__(self):
        pass

    # Returns a binary image of a colored region in an image using thresholding
    # color_thresholds has shape (2,3)
    # color_thresholds[0,i] = min value for color i
    # color_thresholds[1,i] = max value for color i
    def get_blob_threshold(self, img, color_thresholds):
        assert color_thresholds.shape == (2, 3)
        return cv2.inRange(img, color_thresholds[0, :], color_thresholds[1, :])

    # Given a binary image of a blob calculates its centre in pixel coordinates
    def get_blob_coordinates(self, blob):
        # Apply dilate to make the region larger
        #kernel = np.ones((5, 5), np.uint8)
        #blob = cv2.dilate(blob, kernel, iterations=3)
        # Use moments to calculate centre of the blob
        centre = self.centre_of_mass_from_blob(blob)
        # self.circle_centre_from_blob(centre, blob)

        return centre

    def centre_of_mass_from_blob(self, blob):
        # Approximate c.o.m. using moments
        M = cv2.moments(blob)
        approx_cx = int(M['m10'] / M['m00'])
        approx_cy = int(M['m01'] / M['m00'])
        c_o_m = np.array([approx_cx, approx_cy])
        return c_o_m

    def circle_centre_from_blob(self, blob):
        def point_direction(centre, d, maxR):
            epsilon = 5
            minR = 0
            ansX = centre[0]
            ansY = centre[1]
            ansR = 0
            while maxR - minR > epsilon:
                testR = (maxR + minR)/2
                targX = int(np.floor(centre[0] + np.cos(d)*testR))
                targY = int(np.floor(centre[1] - np.sin(d)*testR))
                if blob[targY, targX] != 0:
                    minR = testR
                    ansR = testR
                    ansX = targX
                    ansY = targY
                else:
                    maxR = testR
            return (ansR, np.array([ansX, ansY]))

        approx_c = self.centre_of_mass_from_blob(blob)

        xs, ys = np.nonzero(blob)
        circle_points = np.array([xs, ys]).transpose()
        distances = np.array([np.linalg.norm(approx_c-p) for p in circle_points])


        max_dist = np.max(distances)

        directions = np.linspace(0, 2*np.pi, 9+1)
        directions = directions[0:-1]
        radii = [point_direction(approx_c, d, max_dist) for d in directions]
        radii.sort(key=lambda tup: tup[0])
        centre = mu.find_circle(radii[-1][1], radii[-2][1], radii[-3][1])
        cv2.waitKey(10000)
        return centre


# test the class
def main():
    ce = coordinates_extractor()
    red_thresholds = np.array([[0, 0, 100], [10, 10, 255]])
    img = cv2.imread('image_copy.png', cv2.IMREAD_COLOR)
    blob = ce.get_blob_threshold(img, red_thresholds)
    print(ce.get_blob_coordinates(blob))

if __name__ == '__main__':
    main()
