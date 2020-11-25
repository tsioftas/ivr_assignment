import cv2
import numpy as np

class coordinates_extractor:
    """Class used to extract object pixel coordinates from images"""

    def __init__(self):
        pass


    def get_blob_threshold(self, img, color_thresholds):
        """Returns a binary image of a colored region in an image using thresholding.

        color_thresholds has shape (2,3)
        color_thresholds[0,i] = min value for color i
        color_thresholds[1,i] = max value for color i
        """
        assert color_thresholds.shape == (2, 3)
        return cv2.inRange(img, color_thresholds[0, :], color_thresholds[1, :])


    def get_blob_coordinates(self, blob):
        """Given a binary image of a blob calculates its centre in pixel coordinates.

        Returns None if there are no white pixels in the binary image.
        """
        # Approximate centre of mass using moments
        M = cv2.moments(blob)
        try:
            approx_cx = int(M['m10'] / M['m00'])
            approx_cy = int(M['m01'] / M['m00'])
        except Exception as _:
            return None
        c_o_m = np.array([approx_cx, approx_cy])
        return c_o_m
