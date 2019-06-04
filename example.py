## Existing libraries
import cv2
import numpy as np
import os, sys, time

## Custom
from KinectReader import kinect_reader
from helpers import *

kr = kinect_reader()
wait_for_kinect(kr)

## Refresh kinect frames

while True:
	rgb_flag = kr.update_rgb()
	if(rgb_flag):
		frame = kr.color_image
		frame = cv2.resize(frame, None, fx = 0.5, fy = 0.5)
		cv2.imshow('Frame', frame)
		if cv2.waitKey(1) == ord('q'):
			cv2.destroyAllWindows()
			break
