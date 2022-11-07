from concurrent.futures import process
import numpy as np
from numpy import random
from PIL import Image
import matplotlib.pyplot as plt
import cv2 as cv
import os
class Augmentation:
    def __init__(self, img, resize_shape= (512,512), min_hight=100,min_width=100, angle=None, center = None, scale = None ):
        self.img = img
        self.resize_shape = resize_shape
        self.min_hight = min_hight
        self.min_width = min_width
        self.angle = angle
        self.scale = scale
        self.center = center



    # Scale

    def Scaling(self):
        image_shape = self.img.shape
        scalled_img = cv.resize(self.img,(random.choice(range(image_shape[0]//2,image_shape[0])),random.choice(range(image_shape[1]//2,image_shape[1]))))
        self.img = cv.resize(scalled_img,self.resize_shape)
        return self.img

    # Crop
        
    def Cropping(self):
        image_shape = self.img.shape

        upper_row = random.choice(range(0,image_shape[0]//4))
        lower_row = random.choice(range(upper_row+self.min_hight,image_shape[0]))

        start_column = random.choice(range(0,image_shape[1]//4))
        end_column = random.choice(range(start_column+self.min_width,image_shape[1]))

        crop = self.img[upper_row:lower_row, start_column:end_column]
        self.img = cv.resize(crop,self.resize_shape)

        return self.img

    # Flip
    def Flipping(self):
        self.img = cv.flip(self.img, random.choice([0,1,-1]))
        return self.img

    # Padding


    def Padding(self):
        image_shape = self.img.shape
        l = [cv.BORDER_CONSTANT,cv.BORDER_REFLECT,cv.BORDER_REFLECT_101,cv.BORDER_DEFAULT,cv.BORDER_REPLICATE,cv.BORDER_WRAP]
        img_with_padding = cv.copyMakeBorder(self.img, image_shape[0]//random.choice(range(5,16)), image_shape[0]//random.choice(range(5,16)), image_shape[1]//random.choice(range(5,16)), image_shape[1]//random.choice(range(5,16)),l[random.choice(range(0,6))])
        self.img = cv.resize(img_with_padding,self.resize_shape)
        return self.img



    # Rotation


    def Rotation(self):
        (h, w) = self.img.shape[:2]

        if self.angle is None:
            self.angle = random.choice(range(0,361))

        if self.center is None:
            self.center = (w / 2, h / 2)

        self.scale = random.choice(np.linspace(1,2,9))
        # Perform the rotation
        M = cv.getRotationMatrix2D(self.center, self.angle, self.scale) # An affine transformation is transformation which preserves lines and parallelism.
        # These transformation matrix are taken by warpaffine() function as parameter and the rotated image will be returned.
        self.img = cv.warpAffine(self.img, M, (w, h))
        return self.img

    # Brightness


    # Functions for Color augmentation

    # For HSV, hue range is [0,179], saturation range is [0,255], and value range is [0,255].
    # Different software use different scales.
    # So if you are comparing OpenCV values with them, you need to normalize these ranges.

    def Brightness(self):
        value = random.randint(75, 135)  # cause 255 sometimes is too much
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
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
        self.img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return self.img


    # Saturation

    def Saturation(self):
        value = random.randint(75, 135) # cause 200 is sometime too much
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
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
        self.img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return self.img

    # Hue


    def Hue(self):
        value = random.randint(0,179)
        hsv = cv.cvtColor(self.img, cv.COLOR_BGR2HSV)
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
        self.img = cv.cvtColor(final_hsv, cv.COLOR_HSV2BGR)
        return self.img


    # Resize

    def Resize_image(self):
        self.img = cv.resize(self.img,self.resize_shape)
        return self.img


    
# Randomely generating images

def image_generator(img,number_of_images,resize_shape):
    # resize_shape= 256*256
    # img_p= img.copy()
    # print(len(list_of_op))
    processed_img=[]
    for i in range(number_of_images):
        img_p= img.copy()
        temp_o = Augmentation(img_p,resize_shape)
        list_of_op = [temp_o.Scaling,  temp_o.Cropping,temp_o.Padding,temp_o.Flipping, temp_o.Rotation,  temp_o.Resize_image]
        list_of_col = [temp_o.Brightness, temp_o.Saturation, temp_o.Hue]
        no_of_op = random.randint(1,len(list_of_op))
        print(no_of_op)
        
        random.shuffle(list_of_op) 
        selected_op=list_of_op[0:no_of_op]
        selected_op.append(list_of_col[random.randint(0,3)])
        for j in selected_op:
            print(j.__name__)
            img_p = j()
        processed_img.append(img_p)
    
    return processed_img
    
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img = cv.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    # for i in range(len(images)):
    #     plt.figure(figsize=(20,10))
    #     plt.subplot((len(images)//2)+1,3,i+1)
    #     # plt.title(titles[i])
    #     plt.imshow(cv.cvtColor(images[i],cv.COLOR_BGR2RGB))
    #     plt.axis('off')
    #     plt.show()
    return images

def destination(number_of_images,dest_folder_path,source_folder_path,resize_shape):
    images = load_images_from_folder(source_folder_path)
    no = 1
    # number_of_images=int(input('how many images do you want to augment:\n'))
    #image generator
    for img in images:
        final_images = image_generator(img,number_of_images,resize_shape)
        # l = [img,img_hue,img_brightness,img_cropped,img_flipped,img_padding,img_rotated,img_saturation,img_scalled]
        for i in final_images:
            file_name = dest_folder_path + str(no) + ".jpg"
            no += 1
        #     # file_name =  + '.jpg'
            cv.imwrite(file_name,i) 
            # cv.imshow("window",i) # to see image
            # cv.waitKey(0)  # so image disapear in n sec
        print('done')


# Main Function
if __name__ == "__main__":
    
    
    # number_of_images=int(input('how many images do you want to augment:\n'))
    resize_shape = (512,512)
    destination(number_of_images=11, dest_folder_path='C:\\Users\\admin\\Desktop\\DX codes\\computer visison\\destination\\', 
                source_folder_path = 'C:\\Users\\admin\\Desktop\\DX codes\\computer visison\\source\\',resize_shape= resize_shape)




    # Resizing Image
    # img = Resize_image(img,resize_shape)

    # # fliping image
    # img_flipped = Flipping(img)
    # #Scalling an image
    # img_scalled = Scaling(img,resize_shape)
    # #Crop an image
    # img_cropped = Cropping(img,resize_shape)
    # #Add Padding
    # img_padding = Padding(img,resize_shape)
    # #Rotate an image
    # img_rotated = Rotation(img)
    # #Change Brightness in image
    # img_brightness = Brightness(img)
    # img_saturation = Saturation(img)
    # img_hue = Hue(img)

   
