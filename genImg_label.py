# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:56:10 2019

@author: A30367
"""
import os
import numpy as np
import cv2

def cv2Resize(img, wratio=1, hratio=1, inter=cv2.INTER_AREA):
    (h, w) = img.shape[:2]
    dim = (int(w*wratio), int(h*hratio))
    
    resized = cv2.resize(img, dim, interpolation=inter)
    return resized

def cv2Translate(img, x=0, y=0):
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    
    return shifted

def cv2Rotate(img, angle=0, center=None, scale=1.0):
    (h, w) = img.shape[:2]
    if center is None:
        center = (w / 2, h / 2)
    
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(img, M, (w, h))
    
    return rotated

def cv2Brightness(img, a=0, g=0):
    (h, w, ch) = img.shape
    
    src2 = np.zeros((h, w, ch), img.dtype)
    bImg = cv2.addWeighted(img, a, src2, 1-a, g)
    
    return bImg


def genImg(baseImg, numImg=9):
    imgName = baseImg.split('.jpg')[0]
    print('refer image: {}, #gen: {}'.format(imgName, numImg))
    image = cv2.imread(baseImg)
    
    patterns = {'1': {'func': cv2Resize, 'params': {'wratio': 1, 'hratio': 1.2}},
                '2': {'func': cv2Resize, 'params': {'wratio': 1, 'hratio': 1.3, 'inter': cv2.INTER_NEAREST}},
                '3': {'func': cv2Resize, 'params': {'wratio': 1, 'hratio': 1.4, 'inter': cv2.INTER_CUBIC}},
                '4': {'func': cv2Rotate, 'params': {'angle': 10}},
                '5': {'func': cv2Rotate, 'params': {'angle': -10}},
                '6': {'func': cv2Translate, 'params': {'x': 5}},
                '7': {'func': cv2Translate, 'params': {'x': -5}},
                '8': {'func': cv2Brightness, 'params': {'a': 1.2, 'g': 30}},
                '9': {'func': cv2Brightness, 'params': {'a': 0.5, 'g': 70}},
                }

    count = 1
    for i in sorted(patterns.keys()):
        if count > numImg:
            break
        
        outputImg = patterns[i]['func'](image, **patterns[i]['params'])
        if i in ['1', '2', '3']:
            (h, w) = outputImg.shape[:2]
            prefix = imgName.split('_')[0]
            postId = imgName.split('_')[-1]
            outImgName = prefix + '_w' + str(w) + '_h' + str(h) + '_' + postId + '_' + i + '.jpg'
        else:
            outImgName = imgName+'_'+i+'.jpg'
        
        cv2.imwrite(outImgName, outputImg)
        count += 1

def main():
    labelTxt = 'charImage_20190111T133719.subdir.txt'
    labelDict = {}
    dataList = []
    
    # ==== generate label table ==== #
    with open(labelTxt, 'r') as fid:
        for li in fid:
            fsplit = li.split('\t')
            # folder name
            folder = fsplit[0].split('/')[-1]
            labelDict[folder] = fsplit[1]
    
    # ==== generate fake images to achieve the number ==== #
    # ubuntu: os.walk("./"), windows: os.walk(".\\")
    for root, dirs, files in os.walk("./"):
        print('processing folder: {}'.format(root))
        #if root.split('/')[-1] in ['00000407']:
        if len(files) < 10:
            numImg = 10-len(files)
            ftype = files[0].split('.')[-1]
            if ftype in ['jpg']:
                baseImg = os.path.join(root, files[0])
                genImg(baseImg=baseImg, numImg=numImg)
    
    # ==== generate the .csv file ==== #
    for root, dirs, files in os.walk("./"):
        #print root, dirs, files
        for f in files:
            if root[2:] in labelDict.keys():
                label = labelDict[root[2:]]
                fpath = os.path.join(root, f)
                dataList.append(fpath+","+label)

    with open('char407.csv', 'w') as fid:
        fid.writelines("data,label\n")
        for k in dataList:
            fid.writelines(k+"\n")

if __name__ == '__main__':
    main()