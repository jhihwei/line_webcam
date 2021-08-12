# Thank for Adrian Rosebrock, PhD
# https://www.pyimagesearch.com/2019/09/02/opencv-stream-video-to-web-browser-html-page/

import numpy as np
import imutils
import cv2

class MotionDetector:
    def __init__(self, accum_weight=0.5):
        self.accum_weight = accum_weight
        self.bg = None
    
    def update(self, image):
        if self.bg is None:
            self.bg = image.copy().astype("float")
            return
        cv2.accumulateWeighted(image, self.bg, self.accum_weight)

    def detect(self, image, t_Val=25):
        delta = cv2.absdiff(self.bg.astype("uint8"), image)
        thresh = cv2.threshold(delta, t_Val, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        (minX, minY) = (np.inf, np.inf)
        (maxX, maxY) = (-np.inf, -np.inf)

        # if no contours were found, return None
        if len(cnts) == 0:
            return None
		# otherwise, loop over the contours
		
        for c in cnts:
			# compute the bounding box of the contour and use it to
			# update the minimum and maximum bounding box regions
            (x, y, w, h) = cv2.boundingRect(c)
			
            (minX, minY) = (min(minX, x), min(minY, y))
			
            (maxX, maxY) = (max(maxX, x + w), max(maxY, y + h))
		# otherwise, return a tuple of the thresholded image along
		# with bounding box
		
        return (thresh, (minX, minY, maxX, maxY))

    