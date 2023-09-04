# The tools used in the report

import numpy as np
import math
import skimage
import os
import cv2
import shutil
import statistics

import data
import settings


def calculate_entropy():
    """Calculate min max entropy in training set"""
    index_max = len(os.listdir('input')) // len(settings.name_list)
    min_orig = 100.0
    max_orig = 0.0
    for i in range(1, index_max + 1):
        name = str(i).zfill(3)
        for j in range(2, len(settings.name_list)):
            img = cv2.imread('input_face_cache/' + name + settings.name_list[j] + settings.file_format)
            entropy = skimage.measure.shannon_entropy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            min_orig = min(entropy, min_orig)
            max_orig = max(entropy, max_orig)

    min_mask = 100.0
    max_mask = 0.0

    for i in range(1, index_max + 1):
        name = str(i).zfill(3)
        for j in range(2, len(settings.name_list)):
            img = cv2.imread('color_mask_cache/' + name + settings.name_list[j] + settings.file_format)
            entropy = skimage.measure.shannon_entropy(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            min_mask = min(entropy, min_mask)
            max_mask = max(entropy, max_mask)

    print(min_orig, max_orig)
    print(min_mask, max_mask)


def sorter(path_orig, path_res):
    """Replace results with original images
       STEP 1: Drag the prediction results into different directories
       STEP 2: Run this script"""
    for res in os.listdir(path_res):
        for orig in os.listdir(path_orig):
            if orig == res:
                os.remove(path_res + '/' + res)
                shutil.move(path_orig + '/' + orig, path_res + '/' + res)


def calculate_average_value(path):
    """Calculate average grey, r, g, b, var(grey), area of faces"""
    red = green = blue = 0
    var_all = 0.0
    num = 0
    pixel_num = 0
    for file in os.listdir(path):
        # img = cv2.imread(path + '/' + file)
        img = data.FaceRipper(path + '/' + file).face
        # cv2.imwrite('t.jpg', img)
        t = []
        area = 0
        for i in range(256):
            for j in range(256):
                r, g, b = img[i][j]
                if not (b == 0 and g == 0 and r == 0):
                    red += r
                    green += g
                    blue += b
                    area += 1
                    grey = int(b) + int(g) + int(r)
                    t.append(grey / 3)
        var_all += statistics.variance(t)
        num += 1
        pixel_num += area

    print('R: ' + str(red / pixel_num))
    print('G: ' + str(green / pixel_num))
    print('B: ' + str(blue / pixel_num))
    print('GREY: ' + str((red + green + blue) / 3 / pixel_num))
    print('VAR(GREY): ' + str(var_all / num))
    print('AREA: ' + str(pixel_num / 256 / 256 / num))


def calculate_average_value_training_set(path):
    """Calculate average grey, r, g, b, var(grey), area of the faces in training set"""
    red = green = blue = 0
    var_all = 0.0
    num = 0
    pixel_num = 0
    count = 0
    for file in os.listdir(path):
        # XXXXX_orig.png
        if file[-5] != 'g':
            continue
        img = cv2.imread(path + '/' + file)
        # img = data.FaceRipper(path + '/' + file).face
        # cv2.imwrite('t.jpg', img)
        t = []
        area = 0
        for i in range(256):
            for j in range(256):
                r, g, b = img[i][j]
                if not (b == 0 and g == 0 and r == 0):
                    red += r
                    green += g
                    blue += b
                    area += 1
                    grey = int(b) + int(g) + int(r)
                    t.append(grey / 3)
        var_all += statistics.variance(t)
        num += 1
        pixel_num += area
        count += 1
        print(count)

    print('R: ' + str(red / pixel_num))
    print('G: ' + str(green / pixel_num))
    print('B: ' + str(blue / pixel_num))
    print('GREY: ' + str((red + green + blue) / 3 / pixel_num))
    print('VAR(GREY): ' + str(var_all / num))
    print('AREA: ' + str(pixel_num / 256 / 256 / num))


# calculate_average_value('final eval/ratings/good')
# print("=============================")
# calculate_average_value('final eval/ratings/mediocre')
# print("=============================")
# calculate_average_value('final eval/ratings/poor')
# print("=============================")
calculate_average_value_training_set('input_face_cache')
# calculate_average_value('final eval/ratings/test')
