import numpy as np
import cv2
from Preprocess import Preprocess as pre
from Hand import Hand as hand
import sys

import matplotlib.pyplot as plt
import cv2
import imutils

class Axis:

	def __init__(self):

		self.axis = None
		self.object_pts = None
		self.camera_matrix = None
		#self.img_pts = None
		self.calibrated = False

	# given a convex hull, find points that are likely to be the fingers 
	def find_fingers(self, BGR_frame, hull, palmCenter):

		# all points in the hull as a list
		x = [i[0][0] for i in hull]
		y = [i[0][1] for i in hull]

		# find only the fingers
		fingers = []
		for idx in range(len(x)):

			if y[idx] < palmCenter[1] - 50:
				fingers.append((x[idx],y[idx]))

		# group close points together
		fingers = sorted(fingers, key=lambda a: a[0])
		x_fingers = [i[0] for i in fingers]
		y_fingers = [i[1] for i in fingers]

		# group if within 10 pixels of one another, counts as same 
		idx = 0

		grouped = []

		while idx < len(x_fingers):
			
			same_finger_x = []
			same_finger_y = []

			same_finger_x.append(x_fingers[idx])
			same_finger_y.append(y_fingers[idx])
			
			idx += 1
			while idx < len(x_fingers) and abs(x_fingers[idx] - x_fingers[idx-1]) < 50 and abs(y_fingers[idx] - y_fingers[idx-1]) < 50:
				same_finger_x.append(x_fingers[idx])
				same_finger_y.append(y_fingers[idx])
				idx += 1

			max_y = np.argmin(same_finger_y)
			grouped.append((same_finger_x[max_y], same_finger_y[max_y]))
			
			if len(grouped) >= 5:
				break
		return grouped

	# initial calibration -- approximation of camera matrix, and formation
	# of object points based on the initial calibration image 
	def calibrate(self, image, hull, palmCenter):

		# save camera matrix based on image size
		self.camera_matrix = np.identity(3)#np.array([[BGR_frame.shape[0],0,0],[0,BGR_frame.shape[1],0],[0,0,1]])
		self.camera_matrix[0][0] = image.shape[0]
		self.camera_matrix[1][1] = image.shape[1]
		self.camera_matrix[0][2] = image.shape[0]/2
		self.camera_matrix[1][2] = image.shape[1]/2

		# all points in the hull as a list
		x = [i[0][0] for i in hull]
		y = [i[0][1] for i in hull]

		# find only the fingers
		fingers = []
		for idx in range(len(x)):

			if y[idx] < palmCenter[1] - 50:
				fingers.append((x[idx],y[idx]))

		# group close points together
		fingers = sorted(fingers, key=lambda a: a[0])
		x_fingers = [i[0] for i in fingers]
		y_fingers = [i[1] for i in fingers]

		# group if within 10 pixels of one another, counts as same 
		idx = 0

		grouped = []
		while idx < len(x_fingers):
			
			same_finger_x = []
			same_finger_y = []

			same_finger_x.append(x_fingers[idx])
			same_finger_y.append(y_fingers[idx])
			
			idx += 1
			while idx < len(x_fingers) and abs(x_fingers[idx] - x_fingers[idx-1]) < 50 and abs(y_fingers[idx] - y_fingers[idx-1]) < 50:
				same_finger_x.append(x_fingers[idx])
				same_finger_y.append(y_fingers[idx])
				idx += 1

			max_y = np.argmin(same_finger_y)
			grouped.append((same_finger_x[max_y], same_finger_y[max_y]))
			
			if len(grouped) >= 5:
				break

		grouped_x = [grouped[i][0] for i in range(len(grouped))]
		grouped_y = [grouped[i][1] for i in range(len(grouped))]
		
		# form fixed object points
		self.object_pts = np.zeros((6,3))
		#self.img_pts = np.zeros((6,2))
		for idx in range(len(grouped_x)):
			self.object_pts[idx] = (grouped_x[idx], grouped_y[idx], 0)
		 #   self.img_pts[idx] = (grouped_x[idx], grouped_y[idx])
		self.object_pts[-1] = (palmCenter[0], palmCenter[1], 0)
		#self.img_pts[-1] = palmCenter

		self.calibrated = True
		self.form_axis(palmCenter, grouped_x, grouped_y)
		return grouped
		#plt.imshow(image)
		#plt.scatter(grouped_x, grouped_y)

	# get image points in proper order to be passed to PnP
	def get_img_pts(self, grouped, palmCenter):

		grouped_x = [grouped[i][0] for i in range(len(grouped))]
		grouped_y = [grouped[i][1] for i in range(len(grouped))]

		img_pts = np.zeros((6,2))
		for idx in range(len(grouped_x)):
			img_pts[idx] = (grouped_x[idx], grouped_y[idx])
		img_pts[-1] = palmCenter

		return img_pts

	# creates the fixed 3-d axis based on the initial calibration
	def form_axis(self, palmCenter, grouped_x, grouped_y):


		p1 = palmCenter
		px = (int(grouped_x[0]), int(grouped_y[0]))
		py = (int(grouped_x[2]), int(grouped_y[2]))

		# the y axis is through the middle finger and the palm
		v = np.array([py[0] - p1[0], py[1] - p1[1]])
		u = (v/np.linalg.norm(v)).reshape((2,1))

		# orghotonal projection of thumb point onto this line gives the origin
		thumb_v = np.array([px[0] - p1[0], px[1] - p1[1]])
		origin = np.dot(u, u.T).dot(thumb_v)
		origin[0] = origin[0] + p1[0]
		origin[1] = origin[1] + p1[1]

		#dist_coeffs = np.zeros((4,1))
		#(success, rvec, tvec) = cv2.solvePnP(self.object_pts, self.img_pts, self.camera_matrix, dist_coeffs) #flags=cv2.CV_ITERATIVE)
		
		# form axis based on thumb, middle finger, and orthogonal projection of one on the other
		self.axis = np.zeros((4,3))
		
		self.axis[3] = (origin[0], origin[1], 0)
		self.axis[0] = (py[0], py[1], 0)
		self.axis[1] = (px[0], px[1], 0)
		self.axis[2] = (origin[0], origin[1], -100)


	# solvePnP given new image points, project the points, and return
	# the x, y, z points, as well as the new origin
	def get_axis_2d(self, img_pts):
		dist_coeffs = np.zeros((4,1))

		(success, rvec, tvec) = cv2.solvePnP(self.object_pts, img_pts, self.camera_matrix, dist_coeffs) #flags=cv2.CV_ITERATIVE)
		#print success
		#print self.axis
		#print self.camera_matrix
		(new_axis, jacobian) = cv2.projectPoints(self.axis, rvec, tvec, self.camera_matrix, None)

		x_line = (int(new_axis[0][0][0]), int(new_axis[0][0][1]))
		y_line = (int(new_axis[1][0][0]), int(new_axis[1][0][1]))
		z_line = (int(new_axis[2][0][0]), int(new_axis[2][0][1]))
		origin = (int(new_axis[3][0][0]), int(new_axis[3][0][1]))

		return (x_line, y_line, z_line, origin)

	# groups points close to each other as the same finger, takes 
	# the one with the smallest y value (the highest point in image) as fingertip
	def group_points(self, x_points, y_points):

		idx = 0
		grouped = []

		while idx < len(x_points):

			same_finger_x = []
			same_finger_y = []
			same_finger_x.append(x_points[idx])
			same_finger_y.append(y_points[idx])

			idx += 1
			# check for other points close to it, assuming this is sorted by x coord already
			while idx < len(x_points) and abs(x_points[idx] - x_points[idx-1]) < 40	 \
				and abs(y_points[idx] - y_points[idx-1]) < 40:

				same_finger_x.append(x_points[idx])
				same_finger_y.append(y_points[idx])
				idx += 1

			max_y = np.argmin(same_finger_y)
			grouped.append((same_finger_x[max_y], same_finger_y[max_y]))

		return grouped



