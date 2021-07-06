import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import joblib
from scipy.cluster.vq import kmeans, vq

descriptors_float, des_list, image_paths, image_classes, training_names = joblib.load("descriptors_kaze150_dataset300.pkl")


k = 300
voc, variance = kmeans(descriptors_float, k, 1) 
print(variance)

# Calculate the histogram of features and represent them as vector
#vq Assigns codes from a code book to observations.
im_features = np.zeros((len(image_paths), k), "float32")
for i in range(len(image_paths)):
    words, distance = vq(des_list[i][1],voc)
    for w in words:
        im_features[i][w] += 1
        
# # Perform Tf-Idf vectorization
# nbr_occurences = np.sum( (im_features > 0) * 1, axis = 0)
# idf = np.array(np.log((1.0*len(image_paths)+1) / (1.0*nbr_occurences + 1)), 'float32')

# Scaling the words
#Standardize features by removing the mean and scaling to unit variance
#In a way normalization
stdSlr = StandardScaler().fit(im_features)
im_features = stdSlr.transform(im_features)

print(im_features)
 
lda = LDA()
im_features = lda.fit_transform(im_features, image_classes)
print(im_features)


#save the features
joblib.dump((im_features, image_classes), "features300_kaze_300_HSV+YCbCr.pkl", compress=3)   
joblib.dump((training_names, stdSlr, k, voc, lda), "bovw_features300_kaze_300.pkl", compress=3)  