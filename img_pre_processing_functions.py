# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:24:24 2026

@author: maxva

(Idea of creating a function for preinstalling settings and 
easier application comes from AI code)
function are writen by ourself unless told otherwise in the function
"""
import cv2
import numpy as np

def to_grayscale(img):
    """Easy function to apply grayscale"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_hsv(img):
    """"Easy function to apply HSV"""
    return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

def gaussiann_blur(img, kernal_size = (5,5)):
    """This is the gaussianblur function with a standerd kernal size"""
    return cv2.GaussianBlur(img, kernal_size, 0)

def resize_img_for_testing(img, resizing_factor = 0.65):
    """
    Our data's image are so big the dont fit on a screen. So this function resizes
    the img so that the image is fully visible inside a laptop screen
    """
    return cv2.resize(
        img,
        None,
        fx=resizing_factor,
        fy=resizing_factor,
        interpolation=cv2.INTER_AREA # the interpolation INTER_AREA is usefull for downsizing which is our use case
    )

def c_morph(img):
    """This function performs the morphological close transformation
    help from:
    https://docs.opencv.org/4.13.0/d9/d61/tutorial_py_morphological_ops.html 
    """
    kernel = np.ones((5, 5), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

def HSV_threshold(img, lower_value: np.array, upper_value: np.array, invert_img:bool = 1) -> np.array:
    """This function applies the HSV threshold variable. Right now the assumption is that the values need to be inverted because of the way
    we detect the white background"""
    hsv = to_hsv(img)
    # gray = to_grayscale(img)
    # _, mask = cv2.threshold(gray,100,255,cv2.THRESH_BINARY_INV)
    # mask = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
    #         cv2.THRESH_BINARY_INV,11,8)   # test from opencv docs
    # return c_morph(mask)
    mask = cv2.inRange(hsv, lower_value, upper_value)
    mask = cv2.bitwise_not(mask) if invert_img else mask
    return c_morph(mask)

def find_contours(img):
    """This functions performs the find_contour function with our needed parameters"""
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours
