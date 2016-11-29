import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.externals import joblib
filename = "./res/starye_cifry_i_novye_cifry.jpg"
im  = cv2.imread(filename)

# Convert to grayscale and apply Gaussian filtering
im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
im_gray = cv2.GaussianBlur(im_gray, (5,5), 0)
im_gray = cv2.dilate(im_gray, (7,7))
# Threshold image
cv2.imshow("Gray image", im_gray)
ret, im_th = cv2.threshold(im_gray, 120, 255, cv2.THRESH_BINARY_INV)


cv2.imshow("TH", im_th)

# Find contours in the image
i, ctrs, hier = cv2.findContours(im_th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Get rectangles contains each c ontour
rects = [cv2.boundingRect(ctr) for ctr in ctrs]

# Load classifier
clf = joblib.load('./res/SVNClassifier.pkl')

# For each rectangles region, calculate HOG features and predict
# the digit using Linear SVM
for idx, rect in enumerate(rects):
    # Draw the rectangles
    cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 255), 3)
    # Make the rectanglar region around the digit
    leng1 = int(rect[3] * 1.6)
    leng2 = leng1
    pt1 = int(rect[1] + rect[3] // 2 - leng1 // 2)
    pt2 = int(rect[0] + rect[2] // 2 - leng2 // 2)
    pt1 = 0 if pt1 < 0 else pt1
    pt2 = 0 if pt2 < 0 else pt2
    #roi = im_th[rect[1]: rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
    roi = im_th[pt1:pt1 + leng1, pt2: pt2+leng2]
    leng = int(rect[3] * 1.6)
    src = im_th[rect[1]:rect[1] + rect[3], rect[0] : rect[0] + rect[2]]
    src = cv2.copyMakeBorder(src, rect[2] // 4, rect[2] // 4, rect[3] // 4, rect[3] // 4, cv2.BORDER_CONSTANT, 0)
    #im_th[rect]
    # Resize the image

    roi = cv2.resize(src, (8, 8), interpolation=cv2.INTER_AREA)
    roi = cv2.dilate(roi, (3,3))
    X = roi.reshape((1,-1))
    X = X / 255.0 * 16
    nbr =  clf.predict(X)
    cv2.putText(im, str(int(nbr[0])), (rect[0], rect[1]), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 0, 0), 1)
cv2.imshow("resulting Image rectangles ", im)
cv2.waitKey(0)