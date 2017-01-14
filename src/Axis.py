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

		self.axis_2d = None 

		self.prev_fingers = None 

	def innerAngle(self,px1, py1, px2, py2, cx1, cy1):
	
		dist1 = np.sqrt((px1-cx1)**2 + (py1-cy1)**2)
		dist2 = np.sqrt((px2-cx1)**2 + (py2-cy1)**2)
		
		# find closet point to C
		(Ax,Ay,Bx,By,Cx,Cy) = px2, py2, px1, py1, cx1, cy1
		if dist1 >= dist2:
			Bx = px2
			By = py2
			Ax = px1
			Ay = py1
			
		Q1 = Cx - Ax
		Q2 = Cy - Ay
		P1 = Bx - Ax
		P2 = By - Ay
		
		
		A = np.arccos((P1*Q1 + P2*Q2) / (np.sqrt(P1**2+P2**2) * np.sqrt(Q1**2+Q2**2)))
		A = A*180/np.pi
		return A

	# given a convex hull, find points that are likely to be the fingers 
	def find_fingers(self, BGR_frame, hull, indices, cnt, palmCenter):

		idx = 0
		defects = cv2.convexityDefects(cnt, indices)

		points = []
		for i in range(defects.shape[0]):
			s,e,f,d = defects[i,0]
			p1 = tuple(cnt[s][0])
			p2 = tuple(cnt[e][0])
			p3 = tuple(cnt[f][0])

			x,y,w,h = cv2.boundingRect(hull[0])

			center = (x + float(w)/2, y + float(h)/2)
		
			angle = np.arctan2(center[1] - p1[1], center[0] - p1[0]) * 180 / np.pi
			inAngle = self.innerAngle(p1[0], p1[1], p2[0], p2[1], p3[0], p3[1])
			length = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2)
			
			if (angle > -30 and angle < 160) and (abs(inAngle) > 20) and (abs(inAngle) < 120) and (length > 0.1 * h):
				points.append(p1)
				points.append(p2)

		grouped = []
		x_fingers = [points[i][0] for i in range(len(points))]
		y_fingers = [points[i][1] for i in range(len(points))]

		while idx < len(points):
			
			same_finger_x = []
			same_finger_y = []

			same_finger_x.append(x_fingers[idx])
			same_finger_y.append(y_fingers[idx])
			
			idx += 1
			while idx < len(x_fingers) and abs(x_fingers[idx] - x_fingers[idx-1]) < 30:
				same_finger_x.append(x_fingers[idx])
				same_finger_y.append(y_fingers[idx])
				idx += 1

			max_y = np.argmin(same_finger_y)
			grouped.append((same_finger_x[max_y], same_finger_y[max_y]))

		grouped = sorted(grouped, key=lambda xy: xy[0])
		   
		grouped_x = [grouped[i][0] for i in range(len(grouped))]
		grouped_y = [grouped[i][1] for i in range(len(grouped))]

		#print grouped
		
		self.prev_fingers = grouped
		return grouped

	# initial calibration -- approximation of camera matrix, and formation
	# of object points based on the initial calibration image 
	def calibrate(self, image, hull, indices, cnt, palmCenter):

		# save camera matrix based on image size
		self.camera_matrix = np.array([[995.84054625, 0.,634.00808335],[0.,992.18961999,361.25660867],[0.,0.,1.]])   
		# self.camera_matrix = np.identity(3)#np.array([[BGR_frame.shape[0],0,0],[0,BGR_frame.shape[1],0],[0,0,1]])
		# self.camera_matrix[0][0] = image.shape[0]
		# self.camera_matrix[1][1] = image.shape[1]
		# self.camera_matrix[0][2] = image.shape[0]/2
		# self.camera_matrix[1][2] = image.shape[1]/2

		# all points in the hull as a list
		x = [i[0][0] for i in hull]
		y = [i[0][1] for i in hull]

		# # find only the fingers
		fingers = self.find_fingers(image, hull, indices, cnt, palmCenter)

		# # group close points together
		grouped = sorted(fingers, key=lambda a: a[0])
		

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
		for idx in range(np.min([6,len(grouped_x)])):
			img_pts[idx] = (grouped_x[idx], grouped_y[idx])
		img_pts[-1] = palmCenter

		return img_pts

	# creates the fixed 3-d axis based on the initial calibration
	def form_axis(self, palmCenter, grouped_x, grouped_y):


		p1 = palmCenter
		px = (int(grouped_x[4]), int(grouped_y[4]))
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
		self.axis = np.zeros((8,3))
		
		self.axis[3] = (origin[0], origin[1], 0) #(0,0,0)
		self.axis[0] = (py[0], py[1], 0) #(0,3,0)
		self.axis[1] = (px[0], px[1], 0) #(3,0,0)
		self.axis[2] = (origin[0], origin[1], -100) #(0,0,-3)

		self.axis[4] = (px[0], px[1], -100) #(3,0,-3)
		self.axis[5] = (py[0], py[1], -100)  #(0,-3,-3)
		self.axis[6] = (px[0], py[1], 0) #(3,3,0)
		self.axis[7] = (px[0], py[1], -100) #(3,3,-3)


		  #   1 axis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
	# 2					[0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

	# solvePnP given new image points, project the points, and return
	# the x, y, z points, as well as the new origin
	def get_axis_2d(self, img_pts):
		#dist_coeffs = np.zeros((4,1))
		dist_coeffs = np.array([[-1.91066278e-01,8.25442597e-01,-1.05601400e-03,-5.95083962e-03,-1.14370171e+00]])
		(success, rvec, tvec) = cv2.solvePnP(self.object_pts, img_pts, self.camera_matrix, dist_coeffs) #flags=cv2.CV_ITERATIVE)
		#print success
		#print self.axis
		#print self.camera_matrix
		(new_axis, jacobian) = cv2.projectPoints(self.axis, rvec, tvec, self.camera_matrix, None)

		# x_line = (int(new_axis[0][0][0]), int(new_axis[0][0][1]))
		# y_line = (int(new_axis[1][0][0]), int(new_axis[1][0][1]))
		# z_line = (int(new_axis[2][0][0]), int(new_axis[2][0][1]))
		# origin = (int(new_axis[3][0][0]), int(new_axis[3][0][1]))

		x_line = (int(new_axis[0][0][0]+100), int(new_axis[0][0][1])+200)
		y_line = (int(new_axis[1][0][0]+100), int(new_axis[1][0][1])+200)
		z_line = (int(new_axis[2][0][0]+100), int(new_axis[2][0][1])+200)
		origin = (int(new_axis[3][0][0]+100), int(new_axis[3][0][1])+200)

		pt_5 = (int(new_axis[4][0][0]+100), int(new_axis[4][0][1])+200)
		pt_6 = (int(new_axis[5][0][0]+100), int(new_axis[5][0][1])+200)
		pt_7 = (int(new_axis[6][0][0]+100), int(new_axis[6][0][1])+200)
		pt_8 = (int(new_axis[7][0][0]+100), int(new_axis[7][0][1])+200)

		# check if too different
		thresh = 350
		origin_thresh = 250
		if self.axis_2d:

			if np.linalg.norm(np.array(self.axis_2d[0]) - np.array(x_line)) > thresh:
				x_line = self.axis_2d[0]
			# else:
			# 	print "x ",
			# 	print np.linalg.norm(np.array(self.axis_2d[0]) - np.array(x_line))

			if np.linalg.norm(np.array(self.axis_2d[1]) - np.array(y_line)) > thresh:
				y_line = self.axis_2d[1]
			# else:
			# 	print "y ",
			# 	print np.linalg.norm(np.array(self.axis_2d[1]) - np.array(y_line))

			if np.linalg.norm(np.array(self.axis_2d[2]) - np.array(z_line)) > origin_thresh:
				z_line = self.axis_2d[2]
			# else:
			# 	print "z ",
			# 	print np.linalg.norm(np.array(self.axis_2d[2]) - np.array(z_line))

			if np.linalg.norm(np.array(self.axis_2d[3]) - np.array(origin)) > origin_thresh:
				origin = self.axis_2d[3]
			# else:
			# 	print "origin ",
			# 	print np.linalg.norm(np.array(self.axis_2d[3]) - np.array(origin))

		self.axis_2d = (x_line, y_line, z_line, origin, pt_5, pt_6, pt_7, pt_8)

		# return (x_line, y_line, z_line, origin)
		return self.axis_2d

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



