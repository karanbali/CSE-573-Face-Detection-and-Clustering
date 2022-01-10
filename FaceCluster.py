# Importing Dependencies
import face_recognition as fr
import cv2
import json
import sys
import numpy as np
import math
import re
import os

# Get path of folder containing images (passed as argument)
path = sys.argv[1]
objects = next(os.walk(path))[2]
objects.sort(key=lambda f: int(re.sub('\D', '', f)))

# Number of images contained in the folder
number_of_images = len(objects)

# Get 'k' from the folder's name passed as argument
regex = re.compile(r'\d+')
k = regex.findall(path)
k = int(k[-1])

# Initializing JSON placeholder
json_list = []

# Initializing image encodings placeholder
enc = []

# Initializing  placeholders for storing image number & dimensions
im = []
im_dim = []
np.random.seed(42)

# Initializing haarcascade classifier
face = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')


# Loop all images
for i in range(1, number_of_images+1):

    # Read 'i.jpg' image (MAKE SURE THE PATH DOES NOT CONTAINS 'LEADING SLASH' & is of the form: "./faceCluster_K" OR SEE Notes.txt for more information)
    img = cv2.imread(path+'/'+objects[i-1])

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect all faces in the image
    faces = face.detectMultiScale(gray_img, 1.15, 3)

    # Loop for all detected images
    for (x, y, w, h) in faces:

        # a single element for JSON in a particular format
        # element = {"iname": 'img_' +
        #           str(i)+'.jpg', "bbox": [int(x), int(y), int(w), int(h)]}

        # Face encodings of the particular face (using face_recognition library)
        e = fr.face_encodings(img, [(y, x+w, y+h, x)])

        # append the encoding,image number & dimensions to the respective placeholders 'enc, im, im_dim'
        enc.append(e)
        im.append(i)
        im_dim.append([int(x), int(y), int(w), int(h)])
        # json_list.append(element)


# Euclidean distance b/w 2 points
def Euclidean_distance(a, b):
    return math.sqrt(np.sum((a - b)**2))


# Getting centroids for K-Means++ algorithm
def Centroids_func(data, k):

    # Placeholder for centroids
    centroids = []

    # Randomly choose first centroid
    centroids.append(data[np.random.randint(
        data.shape[0]), :])

    # Loop through for given iterations
    for ind in range(k - 1):

        # Placeholder for distances (used while calculating centroids)
        dist = []

        # Loop through all points
        for i in range(data.shape[0]):
            # i'th point
            point = data[i, :]
            d = sys.maxsize

            # Loop through all centroids
            for j in range(len(centroids)):
                # Calculate euclidean distance b/w centroid & point pair
                temp_dist = Euclidean_distance(point, centroids[j])
                d = min(d, temp_dist)

            # append the best distance
            dist.append(d)

        # Convery to a 'numpy' array
        dist = np.array(dist)

        # Choose farthest point as the next centroid
        next_centroid = data[np.argmax(dist), :]

        # append to placeholder
        centroids.append(next_centroid)
        dist = []

    # Return centroids
    return centroids


# Converting image encodings into a 'Numpy' array
enc = np.array(enc)

# centroids for K-means++
centroids = Centroids_func(enc, k)


# Calculate Distance b/w points & centroids
def Centroids_distance(x, y, eu):
    # Placeholder for distance matrix (b/w points & centroids)
    gap = []

    # Loop through all centroids & points & calculate distance b/w a given pair
    for i in range(len(x)):
        for j in range(len(y)):
            d = x[i][0]-y[j][0]
            d = np.sum(np.power(d, 2))
            gap.append(d)

    # Reshape placeholder
    gap = np.array(gap)
    gap = np.reshape(gap, (len(x), len(y)))

    # Return distance matrix (b/w points & centroids)
    return gap


# Main K-Means++ Function
def kmeans(x, k, cent, iter):

    # Centroids
    centroids = cent

    # Matrix of distance b/w centroids & points
    dist_matrix = Centroids_distance(x, centroids, 'euclidean')

    # Get the nearest centroid (class) for the image
    image_class = np.array([np.argmin(d) for d in dist_matrix])

    # Loop through the number of iterations
    for i in range(iter):

        # Loop to update centroids
        centroids = []
        for j in range(k):

            # Get all encodings for particular class -> Add & find the mean to get a new centroid
            new_cent = x[image_class == j]
            ms = 0
            for l in range(len(new_cent)):
                ms += new_cent[l]

            # Divide to get the mean as a new centroid
            new_cent = np.divide(ms, len(new_cent))

            # append new centroid to the placeholder
            centroids.append(new_cent)

        # Matrix of distance b/w new centroids & points
        dist_matrix = Centroids_distance(x, centroids, 'euclidean')

        # Get the nearest centroid (class) for the image
        image_class = np.array([np.argmin(d) for d in dist_matrix])

    # Return optimal image labels (class)
    return image_class


# Calling main K-Means++ function
km = kmeans(enc, k, centroids, 20)

# Initializing JSON placeholder
json_list = []

# Loop through give iterations
for i in range(k):

    # Initializing JSON placeholders for JSON entry & image clusters-view
    el = []
    hc = []

    # Loop through all images
    for j, val in enumerate(km):

        # 'If' statement to select all images with matching labels(class) & i'th label(class)
        if val == i:

            # a single element for JSON in a particular format
            el.append(objects[j])

            # Read j'th image
            ij = cv2.imread(path+'/'+objects[j])

            # Bounding box dimensions for the face in the particular image
            x, y, w, h = int(im_dim[j][0]), int(
                im_dim[j][1]), int(im_dim[j][2]), int(im_dim[j][3])

            # Crop the face
            ij = ij[y:y+h, x:x+w]

            # Resize the cropped face
            ij = cv2.resize(ij, (80, 80), interpolation=cv2.INTER_NEAREST)

            # append to the placeholder
            hc.append(ij)

    # Horizontally concatenate the i'th cluster images
    h_img = cv2.hconcat(hc)

    # Saving i'th cluster images
    cv2.imwrite('cluster_'+str(i)+'.jpg', h_img)

    # Code to view the i'th cluster images has been commented out intentionally.
    # If you want to see the i'th cluster images than please uncomment the code given below.
    """
    cv2.imshow('Horizontal', h_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    """

    # a single element for JSON in a particular format
    elem = {"cluster_no": i, "elements": el}

    # append to the placeholder
    json_list.append(elem)

# Saving "clusters.json"
output_json = "clusters.json"
with open(output_json, 'w') as f:
    json.dump(json_list, f)
