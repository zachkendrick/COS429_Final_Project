"""
Authors: Zachary Kendrick, Chloe Song, Katy Ho
Class: Princeton University COS429
Project: Augmented Reality

The Hand class provides functions for 
determining different components of a hand.

"""

import cv2
import numpy as np
import sys

class Hand:

    @staticmethod
    def getConvexHull(skin_frame):
        """Finds the convex hull from the largest contour of a binary image
        Parameters:
            skin_frame: binary mask of skin
        Output:
            cnt: list of points that define the contour
            hull: list of points that defines the convex hull of the contour
        """

        _, contours, hierarchy = cv2.findContours(skin_frame, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnt = Hand.getLargestContour(contours)
        cnt = cv2.approxPolyDP(cnt, 0.0005*cv2.arcLength(cnt,True), True)
        hull = cv2.convexHull(cnt)
        return cnt, hull
    

    @staticmethod
    def getLargestContour(contours):
        """Finds the largest contour by area among a list of contours
        Parameters:
            contours: list of contours
        Output:
            returns contour with largest area in image
        """

        max_area=0
        ci=0
        for i in range(len(contours)):
                cnt=contours[i]
                area = cv2.contourArea(cnt)
                if(area>max_area):
                    max_area=area
                    ci=i
        return contours[ci]


    # Credit for this algorithm goes to the paper which can be found at the
    # description in this link: https://www.youtube.com/watch?v=xML2S6bvMwI
    # WARNING: SUPER SLOW!!!!
    # This code was taken from a project. TODO: FIND CITATION
    @staticmethod
    def centerOfPalm(cnt):
        """Finds the center of the palm on a hand 
        Parameters:
            cnt: list of contour points
        Output:
            tuple of x,y coordinates of the center of the palm
        """

        scaleFactor = 0.3
        shrunk = np.array(cnt * scaleFactor, dtype=np.int32)
        tx, ty, w, h = cv2.boundingRect(shrunk)
        maxPoint = None
        maxRadius = 0
        for x in xrange(w):
            for y in xrange(h):
                rad = cv2.pointPolygonTest(shrunk, (tx + x, ty + y), True)
                if rad > maxRadius:
                    maxPoint = (tx + x, ty + y)
                    maxRadius = rad
        if maxPoint is None: return
        realCenter = np.array(np.array(maxPoint) / scaleFactor, dtype=np.int32)
        error = int((1 / scaleFactor) * 1.5)
        maxPoint = None
        maxRadius = 0
        for x in xrange(realCenter[0] - error, realCenter[0] + error):
            for y in xrange(realCenter[1] - error, realCenter[1] + error):
                rad = cv2.pointPolygonTest(cnt, (x, y), True)
                if rad > maxRadius:
                    maxPoint = (x, y)
                    maxRadius = rad
        return maxPoint
        