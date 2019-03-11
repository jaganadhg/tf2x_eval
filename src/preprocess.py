#!/usr/bin/env python
import matplotlib.image as mpimg

import cv2, os
import numpy as np

IMAGE_WIDTH = 320
IMAGE_HEIGHT = 160
IMAGE_CHANNELS = 3

def load_image(data_dir, image_file):
    """
    Load RGB images from a file
    """
    return mpimg.imread(os.path.join(data_dir, image_file.strip()))


def crop(image):
    """
    Crop the image (removing the sky at the top and the car front at the bottom)
    """
    crop_height=180
    crop_width=450
    height = image.shape[0]
    width = image.shape[1]
    
    #return image[60:-25, :, :] # remove the sky and the car front
    y_start=100
    x_start=int(width/2)-int(crop_width/2)
    
    return image[y_start:y_start+crop_height,x_start:x_start+crop_width]


def resize(image):
    """
    Resize the image to the input shape used by the network model
    """
    return cv2.resize(image, (IMAGE_WIDTH, IMAGE_HEIGHT), cv2.INTER_AREA)

def rgb2yuv(image):
    """
    Convert the image from RGB to YUV (This is what the NVIDIA model does)
    """
    return cv2.cvtColor(image, cv2.COLOR_RGB2YUV)


def preprocess(image):
    """
    Combine all preprocess functions into one
    """
    image = crop(image)
    image = resize(image)
    image = rgb2yuv(image)
    return image

def batch_generator(data_dir, image_paths, steering_angles, batch_size):
    """
    Generate training image give image paths and associated steering angles
    """
    images = np.empty([batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS])
    steers = np.empty(batch_size)
    while True:
        i = 0
        for index in np.random.permutation(image_paths.shape[0]):
            #print(index)
            center = image_paths.iloc[index]
            steering_angle = steering_angles.iloc[index]
            # argumentation
            #if is_training and np.random.rand() < 0.6:
            #    image, steering_angle = augument(data_dir, center, left, right, steering_angle)
            #else:
            image = load_image(data_dir, center) 
            # add the image and steering angle to the batch
            images[i] = preprocess(image)
            steers[i] = steering_angle
            i += 1
            if i == batch_size:
                break
        yield images, steers
