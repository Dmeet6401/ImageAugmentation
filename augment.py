# Aim of this py file is create augmented image of given image randomly.

import numpy as np
from numpy import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv


# Functions for Position augmentation
def Scaling(img,resize_shape):
    image_shape = img.shape
    scalled_img = cv.resize(img,(random.choice(range(image_shape[0]//2,image_shape[0])),random.choice(range(image_shape[1]//2,image_shape[1]))))
    resize_512x512 = cv.resize(scalled_img,resize_shape)
    return resize_512x512

def Cropping(img,resize_shape,min_hight=100,min_width=100,):
    image_shape = img.shape

    upper_row = random.choice(range(0,image_shape[0]//2))
    lower_row = random.choice(range(upper_row+min_hight,image_shape[0]))

    start_column = random.choice(range(0,image_shape[1]//2))
    end_column = random.choice(range(start_column+min_width,image_shape[1]))

    crop = img[upper_row:lower_row, start_column:end_column]
    resize_crop_512x512 = cv.resize(crop,resize_shape)

    return resize_crop_512x512

def Flipping(img):
    return cv.flip(img,random.choice([0,1,-1]))

def Padding(img,resize_shape):
    image_shape = img.shape
    l = [cv.BORDER_CONSTANT,cv.BORDER_REFLECT,cv.BORDER_REFLECT_101,cv.BORDER_DEFAULT,cv.BORDER_REPLICATE,cv.BORDER_WRAP]
    img_with_padding = cv.copyMakeBorder(img, image_shape[0]//random.choice(range(5,16)), image_shape[0]//random.choice(range(5,16)), image_shape[1]//random.choice(range(5,16)), image_shape[1]//random.choice(range(5,16)),l[random.choice(range(0,6))])
    resize_img_with_padding_512x512 = cv.resize(img_with_padding,resize_shape)
    return resize_img_with_padding_512x512



def Rotation(img, angle=None, center = None, scale = None):
    (h, w) = img.shape[:2]

    if angle is None:
        angle = random.choice(range(0,361))

    if center is None:
        center = (w / 2, h / 2)

    scale = random.choice(np.linspace(1,2,9))
    # Perform the rotation
    M = cv.getRotationMatrix2D(center, angle, scale) # An affine transformation is transformation which preserves lines and parallelism.
    # These transformation matrix are taken by warpaffine() function as parameter and the rotated image will be returned.
    rotated = cv.warpAffine(img, M, (w, h))
    return rotated

# Functions for Color augmentation

# For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
# Different software use different scales.
# So if you are comparing OpenCV values with them, you need to normalize these ranges.

def Brightness(img):
    value = random.randint(0,200)  # cause 255 sometimes is too much
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    if value>=0:
        lim = 255 - value
        v[v > lim] = 255
        v[v <= lim] += value
    else:
        lim = 0 - value
        v[v < lim] = 0
        v[v >= lim] -= value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

def Saturation(img):
    value = random.randint(0,200) # cause 200 is sometime too much
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    if value>=0:
        lim = 255 - value
        s[s > lim] = 255
        s[s <= lim] += value
    else:
        lim = 0 - value
        s[s < lim] = 0
        s[s >= lim] -= value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

def Hue(img):
    value = random.randint(0,180)
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)

    if value>=0:
        lim = 179 - value
        h[h > lim] = 179
        h[h <= lim] += value
    else:
        lim = 0 - value
        h[h < lim] = 0
        h[h >= lim] -= value

    final_hsv = cv.merge((h, s, v))
    img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
    return img

def Resize_image(img,resize_shape):
    resized_img = cv.resize(img,resize_shape)
    return resized_img


# Main Function
if __name__ == "__main__":
    img = cv.imread('C:\E\DXAssignments\keras\Augmentation\\fruit.jpg')
    resize_shape = (512,512)
    # Resizing Image
    img = Resize_image(img,resize_shape)

    # fliping image
    img_flipped = Flipping(img)

    #Scalling an image
    img_scalled = Scaling(img,resize_shape)

    #Crop an image
    img_cropped = Cropping(img,resize_shape)

    #Add Padding
    img_padding = Padding(img,resize_shape)

    #Rotate an image
    img_rotated = Rotation(img)

    #Change Brightness in image
    img_brightness = Brightness(img)
    img_saturation = Saturation(img)
    img_hue = Hue(img)

    l = [img,img_hue,img_brightness,img_cropped,img_flipped,img_padding,img_rotated,img_saturation,img_scalled]
    no = 23
    for i in l:
        file_name = "C:\E\DXAssignments\keras\Augmentation\img\\"+ str(no) + ".jpg"
        no += 1
        # file_name =  + '.jpg'
        cv.imwrite(file_name,i)
        # cv.imshow("window",i) # to see image  
        # cv.waitKey(0)  # so image disapear in n sec
    print('done')