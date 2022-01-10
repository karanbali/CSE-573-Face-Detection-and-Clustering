# Importing Dependencies
import cv2
import json
import sys
import os
import re

# Get path of folder containing images (passed as argument)
path = sys.argv[1]
objects = next(os.walk(path + '/images'))[2]
objects.sort(key=lambda f: int(re.sub('\D', '', f)))

# Number of images contained in the folder
number_of_images = len(objects)


# Initializing JSON placeholder
json_list = []

# Initializing haarcascade classifier
face = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')

# Loop all images
for i in range(1, number_of_images+1):

    # Read 'i.jpg' image  (MAKE SURE THE PATH DOES NOT CONTAINS 'LEADING SLASH' & is of the form: "./Validation folder" OR SEE "Notes.txt" for more information)
    img = cv2.imread(path+'/images/'+objects[i-1])

    # Convert image to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect all faces in the image
    faces = face.detectMultiScale(gray_img, 1.15, 3)

    # Loop for all detected images
    for (x, y, w, h) in faces:

        # a single element for JSON in a particular format
        element = {"iname": objects[i-1],
                   "bbox": [int(x), int(y), int(w), int(h)]}

        # Code to view & save the detected face in the image has been commented out intentionally.
        # If you want to see the detected face in the image than please uncomment the code given below.
        # Please make sure to MAKE a NEW FOLDER ("Model_Files") in the same directory.

        """
        img_bbox = cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imwrite('./Model_Files/'+objects[i-1], img_bbox)
        #cv2.imshow("Faces found", img)
        # cv2.waitKey(0)
        """

        # append to the JSON placeholder
        json_list.append(element)


# Saving "results.json"
output_json = "results.json"
with open(output_json, 'w') as f:
    json.dump(json_list, f)
