# -*- coding: utf-8 -*-
"""
Created on Wed Aug 17 10:46:54 2022

@author: lenovo
"""

import mediapipe as mp
from mediapipe.python.solutions import pose as mp_pose
import cv2

def pose_detector(im, img_info, tlwhs, team, mp_drawing):
    k = 0
    for tlwh in tlwhs:
        if team[k] == 0: 
            color1 = (34,139,34)
            color2 = (0,255,0)
        else: 
            color1 = (0,0,255)
            color2 = (0,255,255)
        x1 = tlwh[0] - tlwh[2]*0.2
        x2 = tlwh[0] + tlwh[2] + tlwh[2]*0.2
        y1 = tlwh[1] - tlwh[3]*0.2
        y2 = tlwh[1] + tlwh[3] + tlwh[3]*0.2
        if x1<0: x1=0
        if y1<0: y1=0
        if x2>im.shape[1]: x2=im.shape[1]
        if y2>im.shape[0]: y2=im.shape[0]
        with mp_pose.Pose(static_image_mode=False,
                          smooth_landmarks=True, 
                          min_detection_confidence=0.5, 
                          min_tracking_confidence=0.5, 
                          model_complexity=2) as pose:
            #Media pose prediction
            pose_results = pose.process(cv2.cvtColor(img_info['raw_img'][int(y1):int(y2), int(x1):int(x2), :],cv2.COLOR_BGR2RGB))
                        
            mp_drawing.draw_landmarks(im[int(y1):int(y2), int(x1):int(x2), :], pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=color1, thickness=2, circle_radius=2), 
                                      mp_drawing.DrawingSpec(color=color2, thickness=2, circle_radius=2))
        k = k + 1
    return im