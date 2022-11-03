import math
import json
import os.path
import random
import pprint
import numpy as np
import cv2

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return cv2.imread(path, 0)
    else:
        return cv2.imread(path)

'''
image_path = ('AdverseBiNet/train_data/h-dibco2014/H01.png')

# load_image
input_img = imread(image_path)
print(input_img.shape)
w = int(input_img.shape[1])
print(w)
w2 = int(w/2)
img_A = input_img[:, 0:w2]
print(img_A)
img_B = input_img[:, w2:w]
print(img_A.shape)
print(img_B.shape)

# preprocess A and B

img_A = cv2.resize(img_A, (256,256))
img_B = cv2.resize(img_B, (256,256))
print(img_A.shape)
h1 = int(np.ceil(np.random.uniform(1e-2, 286-256)))
w1 = int(np.ceil(np.random.uniform(1e-2, 286-256)))
print(h1, w1)
img_A = img_A[h1:h1+256, w1:w1+256]
img_B = img_B[h1:h1+256, w1:w1+256]
if np.random.random() > 0.5:
    img_A = np.fliplr(img_A)
    img_B = np.fliplr(img_B)
print(img_A.shape)
print(img_B.shape)

img_A = img_A/127.5 - 1.
img_B = img_B/127.5 - 1.

img_AB = np.concatenate((img_A, img_B), axis=2)
# img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
print(img_AB)
print(img_AB.shape)
'''

def load_data(image_path, flip=True, is_test=False):
    img_A, img_B = load_image(image_path)
    img_A, img_B = preprocess_A_and_B(img_A, img_B, flip=flip, is_test=is_test)

    img_A = img_A/127.5 - 1.
    img_B = img_B/127.5 - 1.

    img_AB = np.concatenate((img_A, img_B), axis=2)
    # img_AB shape: (fine_size, fine_size, input_c_dim + output_c_dim)
    return img_AB

def load_image(image_path):
    input_img = imread(image_path)
    w = int(input_img.shape[1])
    w2 = int(w/2)
    img_A = input_img[:, 0:w2]
    img_B = input_img[:, w2:w]

    return img_A, img_B

def preprocess_A_and_B(img_A, img_B, load_size=286, fine_size=256, flip=True, is_test=False):
    if is_test:
        img_A = cv2.resize(img_A, [fine_size, fine_size])
        img_B = cv2.resize(img_B, [fine_size, fine_size])
    else:
        img_A = cv2.resize(img_A, [load_size, load_size])
        img_B = cv2.resize(img_B, [load_size, load_size])

        h1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        w1 = int(np.ceil(np.random.uniform(1e-2, load_size-fine_size)))
        img_A = img_A[h1:h1+fine_size, w1:w1+fine_size]
        img_B = img_B[h1:h1+fine_size, w1:w1+fine_size]

        if flip and np.random.random() > 0.5:
            img_A = np.fliplr(img_A)
            img_B = np.fliplr(img_B)

    return img_A, img_B

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return cv2.imread(path, flatten = True).astype(np.float)
    else:
        return cv2.imread(path).astype(np.float)


def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return cv2.imwrite(path, merge(images, size))

def center_crop(image,size, resize_w=True):
    h,w,c = image.shape
    if size > min(h,w):
        return image
    crop_w = size
    crop_h = size
    mid_x, mid_y = w//2,h//2
    offset_x, offset_y = crop_w//2,crop_h//2
    crop_img = image[mid_y-offset_y:mid_y+offset_y,mid_x-offset_x:mid_x+offset_x]
    return crop_img


def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.
