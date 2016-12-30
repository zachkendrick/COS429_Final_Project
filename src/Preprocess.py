"""
Authors: Zachary Kendrick, Chloe Song, Katy Ho
Class: Princeton University COS429
Project: Augmented Reality

The Preprocessor class provides preprocessing 
functionalities for hand segmentation. It also
serves as a simple wrapper for many existing 
openCV functions.

"""


import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt

class Preprocess:

    rect = []            # the rectangle coordinates of the sampled HSV patch
    cropping = False     # is the frame being cropped
    drag = False         # is the mouse being dragged to expand the roi rectangle
    roiSelected = False  # has the roi been selected

    # threshold boundaries for HSV space skin segmentation
    lower = np.array([0,0,0], dtype = "uint8")
    upper = np.array([0,0,0], dtype = "uint8")


    @staticmethod
    def deNoise(frame):
        """Returns a frame with a gaussian blur applied to it.
        Parameters:
            frame: opencv frame
        Output:
            openCV frame with gaussian blur
        """

        return cv2.GaussianBlur(frame,(5,5),0)


    @staticmethod
    def im2Gray(BGR_frame):
        """Returns a single channel greyscale frame
        Parameters:
            BGR_frame: opencv 3 channel BGR frame
        Output:
            greyscale opencv frame
        """
        return cv2.cvtColor(BGR_frame, cv2.COLOR_BGR2GRAY)


    @staticmethod
    def edges(frame):
        """Runs canny edge detection on image
        Parameters:
            frame: opencv frame
        Output:
            bitmap of edges in image
        """

        # constants determine lower/upper bounds for hysteresis thresholding
        return cv2.Canny(frame, 25, 200)


    @staticmethod
    def thresholdHand(gray_frame):
        """Thresholds an image using OTSU bimodal thresholding
        Parameters:
            gray_frame: opencv gray scale frame
        Output:
            binary mask image of thresholded region 
        """

        # constants determine lower/upper bounds for hysteresis thresholding
        ret, thresh = cv2.threshold(gray_frame,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh


    @staticmethod
    def getHandROI(BGR_frame):
        """Allows the user to select an ROI and converts it to HSV space
        Parameters:
            BGR_frame: opencv 3 channel BGR frame
        Output:
            selected roi region converted to HSV space
        """

        # if the ROI has already been selected return right away
        if Preprocess.roiSelected:
            return None
        copy_frame = BGR_frame.copy()

        # mouse callback function
        cv2.setMouseCallback("frame", Preprocess.click_and_crop)
        
        # executes when mouse if being dragged to select ROI
        if len(Preprocess.rect) == 2 and Preprocess.drag:
            if Preprocess.rect[0][0] != Preprocess.rect[1][0] and Preprocess.rect[0][1] != Preprocess.rect[1][1]:
                roi = copy_frame[Preprocess.rect[0][1]:Preprocess.rect[1][1], Preprocess.rect[0][0]: Preprocess.rect[1][0]]
                cv2.imshow("ROI", roi)
                cv2.rectangle(BGR_frame, Preprocess.rect[0], Preprocess.rect[1], (0, 255, 0), 1)
            return None

        # executes after ROI has been selected
        elif len(Preprocess.rect) == 2 and not Preprocess.drag:
            roi = copy_frame[Preprocess.rect[0][1]:Preprocess.rect[1][1], Preprocess.rect[0][0]: Preprocess.rect[1][0]]
            cv2.destroyWindow("ROI")
            cv2.waitKey(5)
            Preprocess.rect = []
            Preprocess.cropping = False
            Preprocess.roiSelected = True
            return roi

        else:
            return None
            
 
    @staticmethod
    def click_and_crop(event, x, y, flags, param):
        """Mouse callback function for selecting an ROI
        Parameters:
            event: the mouse event
            x: x coordinate of mouse
            y: y coordinate of mouse
            flags: openCV flags
            param: extra parameters
        Output:
            Returns nothing
        """

        if event == cv2.EVENT_LBUTTONDOWN:
            Preprocess.rect = [(x, y)]
            Preprocess.cropping = True
            Preprocess.drag = True

        elif event == cv2.EVENT_MOUSEMOVE and Preprocess.drag:
            if(len(Preprocess.rect) == 2):
                Preprocess.rect[1] = (x, y)
            else:
                Preprocess.rect.append((x, y))

        # check to see if the left mouse button was released
        elif event == cv2.EVENT_LBUTTONUP:
            Preprocess.rect[1] = (x, y)
            Preprocess.cropping = False
            Preprocess.drag = False


    @staticmethod
    def updateHSV(roi):
        """Updates HSV space boundaries given an ROI 
        Parameters:
            roi: ROI in BGR
        Output:
            lower: HSV space lower boundary
            upper: HSV space upper boundary
        """
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        lower = np.array([0,0,0], dtype = "uint8")
        upper = np.array([0,0,0], dtype = "uint8")

        # calculate cumulative sum histograms of each channel
        h, s, v = cv2.split(roi)
        h_sum, bins, patches = plt.hist(h.flatten(), 256, range = (1, 255), cumulative = True, fc='b', ec='b', alpha=0.5, label='this image')
        s_sum, bins, patches = plt.hist(s.flatten(), 256, range = (1, 255), cumulative = True, fc='b', ec='b', alpha=0.5, label='this image')
        v_sum, bins, patches = plt.hist(v.flatten(), 256, range = (1, 255), cumulative = True, fc='b', ec='b', alpha=0.5, label='this image')

        # Uncomment to plot cummulative sum histograms
        # plt.show()

        # update the HSV boundary ranges
        lower, upper = Preprocess.change_range(lower, upper, h_sum, 0)
        lower, upper = Preprocess.change_range(lower, upper, s_sum, 1)
        lower, upper = Preprocess.change_range(lower, upper, v_sum, 2)

        return lower, upper


    @staticmethod
    def change_range(lower, upper, cumsum, hsv):
        """Helper function to update the HSV boudary space
        Parameters:
            lower: current HSV space lower boundary
            upper: current HSV space upper boundary
            cumsum: cumulative sum histogram
            hsv: the channel number (0:H, 1:S, 2:V)
        Output:
            lower: updated HSV space lower boundary
            upper: updated HSV space upper boundary
        """

        topPerc = 0.99
        lowPerc = 0.01
        lowBound = lowPerc*max(cumsum)
        upBound = topPerc*max(cumsum)
        for i in range(0, 256):
            if cumsum[i] < lowBound: lower[hsv] = i
            if cumsum[i] < upBound: upper[hsv] = i
        return lower, upper

 
    @staticmethod
    def getSkinMask(lower, upper, BGR_frame):
        """Returns a skin mask defined by HSV thresholds
        Parameters:
            lower: HSV space lower boundary
            upper: HSV space upper boundary
            BGR_frame: openCV 3 channel BGR frame
        Output:
            binary image of skin mask
        """

        frame = cv2.cvtColor(BGR_frame, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(BGR_frame, lower, upper)
        kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 1))
        kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4))
        skinMask = cv2.erode(skinMask, kernel1, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel2, iterations = 2)
        return skinMask

    @staticmethod
    def applyMask(mask, frame):
        """Applies a mask to a given frame
        Parameters:
            mask: binary mask
            frame: openCV frame
        Output:
            original frame with binary mask applied to it
        """

        return cv2.bitwise_and(frame, frame, mask= mask)