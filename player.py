# -*- coding: utf-8 -*-
"""
Created on Tue Aug 16 16:21:57 2022

@author: lenovo
"""

from shapely.geometry import Point, Polygon
import torch
import numpy as np
import cv2


def isplayer(court, outputs):
    for i in range(len(outputs[0]), -1):
        point = (outputs[0][i][1]+outputs[0][i][3]//2, outputs[0][i][0])
        if Point(point).within(court) == False:
            outputs[0] = outputs[0][torch.arange(outputs[0].size(0))!=i]
    return outputs


def drawPlayers(im, tlwhs, ids, team, h):
    points = []
    for tlwh in tlwhs:
        x = tlwh[0]+tlwh[2]*0.5
        y = tlwh[1]+tlwh[3]
        points.append([y,x])
    points = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    new = cv2.perspectiveTransform(points, h)
    for i in range(len(new)):
        if team[i] == 0:color = (0,255,0)
        else: color = (0,0,255)
        im = cv2.circle(im, (new[i][0][1], new[i][0][0]), radius=15, color=color, thickness=3)
        im = cv2.putText(im, str(ids[i]), (int(new[i][0][1]-8), int(new[i][0][0]+8)), cv2.FONT_HERSHEY_SIMPLEX, 
                         color=color, fontScale=0.8, thickness=2)
    return im

def reid(features0, features, ids, team):
    for i in range(len(ids)):
        if ids[i] not in [1,2,3,4,5,6]:
            deffiences = []
            for j in range(len(features0)):
                deffiences.append(np.linalg.norm(features0[j]-features[i]))
            no3 = np.argsort(deffiences)[::-1]
            for no in no3:
                if no+1 not in ids:
                    ids[i] = no+1
    return ids

def repositon(tlwhs, positions, ids):
    for j in range(1,7):
        if j not in ids:
            tlwhs.append(positions[j][:])
            ids.append(j)
            continue
        positions[j][:] = tlwhs[ids.index(j)]
    return tlwhs, ids, positions