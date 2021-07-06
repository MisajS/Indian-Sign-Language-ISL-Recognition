import cv2
import numpy as np
import os
import joblib

# Get the training classes names and store them in a list
# Here we use folder names for class names
# train_path = 'Dataset/Train'
# train_path = 'Dataset/sample/Train'  # Folder Names are Parasitized and Uninfected
train_path = 'D:/Misaj/Mini Project/sample/train'
training_names = os.listdir(train_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
def imglist(path):    
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
    
for training_name in training_names:
    dir = os.path.join(train_path, training_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1

# Create feature extraction and keypoint detector objects
# Create List where all the descriptors will be stored
des_list = []

#sift
sift = cv2.SIFT_create()

for image_path in image_paths:
    print(image_path)
    img = cv2.imread(image_path)
    # resize image
    dim = (200, 200)
    img_gray = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    
    skinG = cv2.GaussianBlur(img_gray, (3,3), 0)
    
    # Skin segmentation using combined HSV and YCbCr models
    img_HSV = cv2.cvtColor(skinG, cv2.COLOR_BGR2HSV)
    # Single Channel mask,denoting presence of colours in the about threshold
    HSV_mask = cv2.inRange(img_HSV, (0, 15, 0), (17, 170, 255))
    img_YCrCb = cv2.cvtColor(skinG, cv2.COLOR_BGR2YCrCb)
    YCrCb_mask = cv2.inRange(img_YCrCb, (0, 135, 85), (255, 180, 135)) 
    YCrCb_mask = cv2.morphologyEx(YCrCb_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    #merge skin detection (YCbCr and hsv)
    global_mask=cv2.bitwise_and(YCrCb_mask,HSV_mask)
    
    # apply a series of erosions and dilations to the mask using an elliptical kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))  
    # skinMask = cv2.erode(skinMask,kernel,iterations = 1)
    skinMask = cv2.dilate(global_mask,kernel,iterations = 1) 
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))  
    skinMask = cv2.erode(skinMask,kernel,iterations = 1)
    
    # blur the mask to help remove noise, then apply the mask to the frame
    # skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(img_gray, img_gray, mask = skinMask)
    
    # skinG = cv2.GaussianBlur(skin, (3,3), 0)
    im = cv2.Canny(skin, 100,200)

    kpts, des = sift.detectAndCompute(im, None)
    des_list.append((image_path, des))   
    if des.any == None:
        break
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[1:]:
    descriptors = np.vstack((descriptors, descriptor))  

#kmeans works only on float, so convert integers to float
descriptors_float = descriptors.astype(float)  


joblib.dump((descriptors_float, des_list, image_paths, image_classes, training_names), "descriptors_sift_dataset300_train.pkl", compress=3) 
 