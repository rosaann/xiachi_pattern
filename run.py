#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 12 15:35:13 2018

@author: zl
"""
import cv2
import numpy as np
import os
import torch
import numpy as np
import torchvision
#rom utils import plot_images
from torchvision import datasets

import visdom
import json
import base64

def checkDefect(imgToBeDetected, patternImg,patternLabel, threshold):
    img_gray = cv2.cvtColor(imgToBeDetected, cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(img_gray, patternImg, cv2.CV_TM_CCOEFF_NORMED)
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(imgToBeDetected, pt, (pt[0] + w, pt[1] + h), (7,249,151), 2)
        
    vis.images(imgToBeDetected, win=p_idx, opts={'title': patternLabel})
    p_idx += 1;

    return res;

def check(imgDir):
    img_src_rgb = cv2.imread(imgDir)
    for patern in data_base:
        res = checkDefect(img_src_rgb, patern['img'],patern['label'], 0)
        
                
def genDataBase():
    base_database_dir1 = "guangdong_round2_train_20181011/单瑕疵图片/"
    data_base = []
    for subPatternDir in base_database_dir1:
        subDir = base_database_dir1 + subPatternDir
        print('subDir : {}'.format(subDir))
        for file in os.listdir(subDir):
            if file.find(".json"):
                json_data = json.load(open(file))
                img_rgb = base64.b64decode(json_data['imageData'])
                shapes = json_data['shapes']          
                for shape in shapes:
                    patern = {}
                    patern['type'] = shape['label']
                    points = shape['points']
                    if len(points) == 4:
                        l = points[0][0]
                        r = points[1][0]
                        t = points[0][1]
                        b = points[2][1]
                        patern['img'] = img_rgb[l:r, t:b]
                        data_base.add(patern)
    
    return data_base
vis = visdom.Visdom(server="http://localhost", port="8888")
baseValidDir = "/guangdong_round1_train2_20180916/瑕疵样本/"
data_base = genDataBase()
print('data_base_len : {}'.format(len(data_base)))
p_idx = 0;
for i_1, subPattern in enumerate( os.listdir(baseValidDir) ):
    if i_1 == 0:
        vis.text(subPattern)
        #接下来得到子目录下的所有图片
        subDir = baseValidDir + subPattern
        for i, file in enumerate( os.listdir(subDir)):
            #开始检查图片
            if i == 0:
                p_idx = 0;
                check(file)
            