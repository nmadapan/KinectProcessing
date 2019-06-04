from __future__ import print_function

from glob import glob
import os, sys, time
import numpy as np
import json
from scipy.interpolate import interp1d
import cv2
import re
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from copy import deepcopy

###########
## PATHS ##
###########
kinect_joint_names_path = 'kinect_joint_names.json'

def json_to_dict(json_filepath):
	if(not os.path.isfile(json_filepath)):
		raise IOError('Error! Json file: '+json_filepath+' does NOT exists!')
	with open(json_filepath, 'r') as fp:
		var = json.load(fp)
	return var

########################
### Kinect Joint IDs ###
########################
## Refer to kinect_joint_names.json for all joint IDs

kinect_joint_names_dict = json_to_dict(kinect_joint_names_path)
kj = kinect_joint_names_dict

## Left hand
left_hand_id = kj['JointType_HandLeft'] # 7
left_elbow_id = kj['JointType_ElbowLeft'] # 5
left_shoulder_id = kj['JointType_ShoulderLeft'] # 4

## Right hand
right_hand_id = kj['JointType_HandRight'] # 11
right_elbow_id = kj['JointType_ElbowRight'] # 9
right_shoulder_id = kj['JointType_ShoulderRight'] # 8

## Torso
torso_id = kj['JointType_SpineBase'] # 0
neck_id = kj['JointType_Neck'] # 2

def wait_for_kinect(kr, timeout = 30):
	'''
	Description:
		* This function waits for the all modalities (RGB, Depth, Body) of the Kinect to connect. It is a blocking function.
	Input arguments:
		* kr is an object of kinect_reader class present in KinectReader module
		* timeout is time in seconds. How long to wait for connection.
	How to call:
		from KinectReader import kinect_reader
		kr = kinect_reader()
		wait_for_kinect(kr)
	'''

	## Initialization
	first_rgb, first_depth, first_body = False, False, False
	init_start_time = time.time()
	print('Connecting to Kinect . ', end = '')

	## Wait for all modules (rgb, depth, skeleton) to connect for timeout secs.
	while True:
		try:
			# Refreshing RGB Frames
			if first_rgb: kr.update_rgb()
			else: first_rgb = kr.update_rgb()
			# Refreshing Depth Frames
			if first_depth: kr.update_depth()
			else: first_depth = kr.update_depth()
			# Refreshing Body Frames
			if first_body: kr.update_body()
			else: first_body = kr.update_body()
			if (first_rgb and first_depth and first_body): break
			time.sleep(0.5)
			print('. ', end = '')
		except Exception as exp:
			print(exp)
			time.sleep(0.5)
			print('. ', end = '')
		if(time.time()-init_start_time > timeout):
			sys.exit('Waited for more than ' + str(timeout) + ' seconds. Exiting')
	print('\nAll Kinect modules connected !!')

def sync_ts(skel_ts, rgb_ts):
	'''
	Description:
		Synchronize two lists of time stamps.
	Input arguments:
		* skel_ts: list of skeleton time stamps. [t1, t2, ..., ta]
		* rgb_ts: list of rgb times tamps. [t1, t2, t3, ..., tb]
	Return:
		* A tuple (apply_on_skel, apply_on_rgb)
			apply_on_skel: What is the corresponding rgb time stamp for every skeleton time stamp.
			apply_on_rgb: What is the corresponding skeleton time stamp for every rgb time stamp.
	'''
	skel_ts = np.reshape(skel_ts, (-1, 1))
	rgb_ts = np.reshape(rgb_ts, (1, -1))
	M = np.abs(skel_ts - rgb_ts)
	apply_on_skel = np.argmin(M, axis = 0)
	apply_on_rgb = np.argmin(M, axis = 1)
	return apply_on_skel.tolist(), apply_on_rgb.tolist()

def draw_body(img = None, img_skel_pts = None, only_upper_body = True, line_color = \
	(255,255,255), thickness = 15, draw_gest_thresh = True, thresh_level = 0.2):
	'''
	Description:
		Draw the skeleton on the image (RGB/Depth), given the pixel coordinates of the skeleton.
	Input arguments:
		* img: np.ndarray. Either RGB (H x W x 3) or Depth (H x W)
		* img_skel_pts: A list of 50 elements. [x0, y0, x1, y1, x2, y2, ...]. (xi,yi) is the pixel coordinate of the ith kinect joint.
		* only_upper_body: If True, only upper body is drawn
		* line_color: The color of the line. It is a tuple of three elements. (B, G, R). B, G and R are the intensity values of blue, green and red channels.
		* thickness: thickness of the line in terms of no. of pixels.
		* thresh_level: What is the threshold level at which gesture will start. 0.2 indicates 20% of the (distance between torso and the neck) above the torso level.
		* draw_gest_thresh: If True, gesture threshold is drawn on the image.
	Return:
		* img: np.ndarray of same size as input image. The image with the skeleton drawn on a copy of the input image.
	'''

	# Return None, if the img or the skeleton points are None
	if(img is None or img_skel_pts is None): return None

	def display_joint(j_start, j_end):
		'''
		Description:
			Draw a line between the given two kinect joints.
		Input arguments:
			j_start: starting kinect joint index
			j_end: ending kinect joint index
		Return:
			True, on sucess. False, on failure.
		'''
		try:
			start = (int(img_skel_pts[2*j_start]), int(img_skel_pts[2*j_start+1]))
			end = (int(img_skel_pts[2*j_end]), int(img_skel_pts[2*j_end+1]))
			cv2.line(img, start, end, line_color, thickness)
			return True
		except Exception as exp:
			return False

	# Head/Neck/Torso
	display_joint(kj['JointType_Head'], kj['JointType_Neck'])
	display_joint(kj['JointType_Neck'], kj['JointType_SpineShoulder'])
	display_joint(kj['JointType_SpineShoulder'], kj['JointType_SpineMid'])
	display_joint(kj['JointType_SpineMid'], kj['JointType_SpineBase'])
	display_joint(kj['JointType_SpineShoulder'], kj['JointType_ShoulderRight'])
	display_joint(kj['JointType_SpineShoulder'], kj['JointType_ShoulderLeft'])
	display_joint(kj['JointType_SpineBase'], kj['JointType_HipRight'])
	display_joint(kj['JointType_SpineBase'], kj['JointType_HipLeft'])

	# Upper left limb
	display_joint(kj['JointType_ShoulderLeft'], kj['JointType_ElbowLeft'])
	display_joint(kj['JointType_ElbowLeft'], kj['JointType_WristLeft'])
	display_joint(kj['JointType_WristLeft'], kj['JointType_HandLeft'])
	# display_joint(kj['JointType_HandLeft'], kj['JointType_HandTipLeft'])
	# display_joint(kj['JointType_WristLeft'], kj['JointType_ThumbLeft'])

	# Upper Right limb
	display_joint(kj['JointType_ShoulderRight'], kj['JointType_ElbowRight'])
	display_joint(kj['JointType_ElbowRight'], kj['JointType_WristRight'])
	display_joint(kj['JointType_WristRight'], kj['JointType_HandRight'])
	# display_joint(kj['JointType_HandRight'], kj['JointType_HandTipRight'])
	# display_joint(kj['JointType_WristRight'], kj['JointType_ThumbRight'])

	# If draw_gest_thresh is True, Draw the gesture threshold.
	if(draw_gest_thresh):
		neck = img_skel_pts[2*kj['JointType_Neck']:2*kj['JointType_Neck']+2]
		base = img_skel_pts[2*kj['JointType_SpineBase']:2*kj['JointType_SpineBase']+2]

		# If neck or base are not detected, don't draw anything
		if(np.isinf(np.sum(neck)) or np.isnan(np.sum(neck))): return None
		if(np.isinf(np.sum(base)) or np.isnan(np.sum(base))): return None

		thresh_x = int(base[0])
		thresh_y = int(thresh_level * (neck[1] - base[1]) + base[1])

		thresh_disp_len = int(thresh_level * (neck[1] - base[1]))

		## Boundary conditions for starting point
		start_x = thresh_x - thresh_disp_len
		if(start_x < 0): start_x = 0
		elif(start_x >= img.shape[1]): start_x = img.shape[1] - 1
		## Boundary conditions for ending point
		end_x = thresh_x + thresh_disp_len
		if(end_x < 0): end_x = 0
		elif(end_x >= img.shape[1]): end_x = img.shape[1] - 1

		start = (start_x, thresh_y)
		end = (end_x, thresh_y)

		cv2.circle(img,(int(thresh_x),int(thresh_y)), 10, (0,0,255), -1)
		cv2.line(img, start, end, (50, 0, 255), thickness)

	## If only_upper_body is True, draw only the upper body. Else, draw the entire body.
	if(not only_upper_body):
		# Lower left limb
		display_joint(kj['JointType_HipLeft'], kj['JointType_KneeLeft'])
		display_joint(kj['JointType_KneeLeft'], kj['JointType_AnkleLeft'])
		display_joint(kj['JointType_AnkleLeft'], kj['JointType_FootLeft'])

		# Lower right limb
		display_joint(kj['JointType_HipRight'], kj['JointType_KneeRight'])
		display_joint(kj['JointType_KneeRight'],kj['JointType_AnkleRight'])
		display_joint(kj['JointType_AnkleRight'], kj['JointType_FootRight'])

	return img
