import cv2
import numpy as np
import os
import pylab as pl
from sklearn.svm import LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
import joblib

# Load the classifier, class names, scaler, number of clusters and vocabulary 
#from stored pickle file (generated during training)
im_features, image_classes = joblib.load("features300_sift_2sample_HSV+YCbCr.pkl")
classes_names, stdSlr, k, voc, lda = joblib.load("bovw_features300_sift_2sample.pkl")


clf_svm = LinearSVC(max_iter=10000)  #Default of 100 is not converging
clf_svm.fit(im_features, np.array(image_classes))

clf_knn = KNeighborsClassifier(n_neighbors=3)
clf_knn.fit(im_features, np.array(image_classes))

clf_reg = LogisticRegression(random_state = 0) 
clf_reg.fit(im_features, np.array(image_classes))

clf_nb_g = GaussianNB()
clf_nb_g.fit(im_features, np.array(image_classes))

clf_nb_b = BernoulliNB()
clf_nb_b.fit(im_features, np.array(image_classes))

clf_rf = RandomForestClassifier(max_depth=19, random_state=0)
clf_rf.fit(im_features, np.array(image_classes))

clf_dt = DecisionTreeClassifier(max_depth=19)
clf_dt.fit(im_features, np.array(image_classes))
# clf_nb_m = MultinomialNB()
# clf_nb_m.fit(im_features, np.array(image_classes))


# Get the path of the testing image(s) and store them in a list
# test_path = 'Dataset/Test'
# test_path = 'Dataset/sample/Test'  # Folder Names are Parasitized and Uninfected
test_path = 'D:/Misaj/Mini Project/sample/test'
#instead of test if you use train then we get great accuracy

testing_names = os.listdir(test_path)

# Get path to all images and save them in a list
# image_paths and the corresponding label in image_paths
image_paths = []
image_classes = []
class_id = 0

#To make it easy to list all file names in a directory let us define a function
def imglist(path):
    return [os.path.join(path, f) for f in os.listdir(path)]

#Fill the placeholder empty lists with image path, classes, and add class ID number
for testing_name in testing_names:
    dir = os.path.join(test_path, testing_name)
    class_path = imglist(dir)
    image_paths+=class_path
    image_classes+=[class_id]*len(class_path)
    class_id+=1
    
# Create feature extraction and keypoint detector objects   
# Create List where all the descriptors will be stored
des_list = []

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
    
# Stack all the descriptors vertically in a numpy array
descriptors = des_list[0][1]
for image_path, descriptor in des_list[0:]:
    descriptors = np.vstack((descriptors, descriptor)) 

# Calculate the histogram of features
#vq Assigns codes from a code book to observations.
from scipy.cluster.vq import vq    
test_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        test_features[i][w] += 1

# Perform Tf-Idf vectorization
nbr_occurences = np.sum( (test_features > 0) * 1, axis = 0)
idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scale the features
#Standardize features by removing the mean and scaling to unit variance
#Scaler (stdSlr comes from the pickled file we imported)
test_features = stdSlr.transform(test_features)

test_features = lda.transform(test_features)



#######Until here most of the above code is similar to Train except for kmeans clustering####

#Report true class names so they can be compared with predicted classes
true_class =  [classes_names[i] for i in image_classes]
# SVM- Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf_svm.predict(test_features)]
#Print the true class and Predictions 
# print ("true_class ="  + str(true_class))
print("Linear SVM")
print ("prediction ="  + str(predictions))
#To make it easy to understand the accuracy let us print the confusion matrix
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix- Linear SVM')
    pl.colorbar()
    pl.show()
accuracy = accuracy_score(true_class, predictions)
p, r, f, s = precision_recall_fscore_support(true_class, predictions, average = 'macro')
print ("accuracy  = ", accuracy)
print ("precision = ", p)
print ("recall    = ", r)
print ("f1_score  = ", f)
cm = confusion_matrix(true_class, predictions)
print (cm)
showconfusionmatrix(cm)



true_class =  [classes_names[i] for i in image_classes]
# KNN- Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf_knn.predict(test_features)]
#Print the true class and Predictions 
# print ("true_class ="  + str(true_class))
print("K Nearest Neighbour")
print ("prediction ="  + str(predictions))
#To make it easy to understand the accuracy let us print the confusion matrix
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix- K Nearest Neighbour')
    pl.colorbar()
    pl.show()
accuracy = accuracy_score(true_class, predictions)
p, r, f, s = precision_recall_fscore_support(true_class, predictions, average = None)
print ("accuracy  = ", accuracy)
print ("precision = ", p)
print ("recall    = ", r)
print ("f1_score  = ", f)
cm = confusion_matrix(true_class, predictions)
print (cm)
showconfusionmatrix(cm)



# Logistic Regression- Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf_reg.predict(test_features)]

#Print the true class and Predictions 
# print ("true_class ="  + str(true_class))
print("Linear Regression")
print ("prediction ="  + str(predictions))
#To make it easy to understand the accuracy let us print the confusion matrix
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Linear Regression')
    pl.colorbar()
    pl.show()
accuracy = accuracy_score(true_class, predictions)
p, r, f, s = precision_recall_fscore_support(true_class, predictions, average = None)
print ("accuracy  = ", accuracy)
print ("precision = ", p)
print ("recall    = ", r)
print ("f1_score  = ", f)
cm = confusion_matrix(true_class, predictions)
print (cm)
showconfusionmatrix(cm)



# NB- Gaussian- Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf_nb_g.predict(test_features)]
#Print the true class and Predictions 
# print ("true_class ="  + str(true_class))
print("Naive Bayes- Gaussian")
print ("prediction ="  + str(predictions))
#To make it easy to understand the accuracy let us print the confusion matrix
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix- Naive Bayes- Gaussian')
    pl.colorbar()
    pl.show()
accuracy = accuracy_score(true_class, predictions)
p, r, f, s = precision_recall_fscore_support(true_class, predictions, average = None)
print ("accuracy  = ", accuracy)
print ("precision = ", p)
print ("recall    = ", r)
print ("f1_score  = ", f)
cm = confusion_matrix(true_class, predictions)
print (cm)
showconfusionmatrix(cm)



# NB_ Binomial- Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf_nb_b.predict(test_features)]
#Print the true class and Predictions 
# print ("true_class ="  + str(true_class))
print("Naive Bayes- Binomial")
print ("prediction ="  + str(predictions))
#To make it easy to understand the accuracy let us print the confusion matrix
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix- Naive Bayes Binomial')
    pl.colorbar()
    pl.show()
accuracy = accuracy_score(true_class, predictions)
p, r, f, s = precision_recall_fscore_support(true_class, predictions, average = None)
print ("accuracy  = ", accuracy)
print ("precision = ", p)
print ("recall    = ", r)
print ("f1_score  = ", f)
cm = confusion_matrix(true_class, predictions)
print (cm)
showconfusionmatrix(cm)



# Decision Tree Classifier- Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf_dt.predict(test_features)]
#Print the true class and Predictions 
# print ("true_class ="  + str(true_class))
print("Decision Tree")
print ("prediction ="  + str(predictions))
#To make it easy to understand the accuracy let us print the confusion matrix
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix- Decision Tree')
    pl.colorbar()
    pl.show()
accuracy = accuracy_score(true_class, predictions)
p, r, f, s = precision_recall_fscore_support(true_class, predictions, average = None)
print ("accuracy  = ", accuracy)
print ("precision = ", p)
print ("recall    = ", r)
print ("f1_score  = ", f)
cm = confusion_matrix(true_class, predictions)
print (cm)
showconfusionmatrix(cm)



# Random Forest- Perform the predictions and report predicted class names. 
predictions =  [classes_names[i] for i in clf_rf.predict(test_features)]
#Print the true class and Predictions 
# print ("true_class ="  + str(true_class))
print("Random Forest")
print ("prediction ="  + str(predictions))
#To make it easy to understand the accuracy let us print the confusion matrix
def showconfusionmatrix(cm):
    pl.matshow(cm)
    pl.title('Confusion matrix- Random Forest')
    pl.colorbar()
    pl.show()
accuracy = accuracy_score(true_class, predictions)
p, r, f, s = precision_recall_fscore_support(true_class, predictions, average = None)
print ("accuracy  = ", accuracy)
print ("precision = ", p)
print ("recall    = ", r)
print ("f1_score  = ", f)
cm = confusion_matrix(true_class, predictions)
print (cm)
showconfusionmatrix(cm)


"""
#For classification of unknown files we can print the predictions
#Print the Predictions 
print ("Image =", image_paths)
print ("prediction ="  + str(predictions))
#np.transpose to save data into columns, otherwise saving as rows
np.savetxt ('mydata.csv', np.transpose([image_paths, predictions]),fmt='%s', delimiter=',', newline='\n')
"""