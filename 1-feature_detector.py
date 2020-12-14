import cv2
import numpy as np

# img1 = cv2.imread('ImagesQuery/project1.jpg', 0)        # 0 is used to import the image in GrayScale
img1 = cv2.imread('Color_Image/B11-GD.jpg', 0)        # 0 is used to import the image in GrayScale
img1 = cv2.resize(img1, (500, 600))
# img2 = cv2.imread('ImagesTrain/p2.jpg', 0)
img2 = cv2.imread('Color_Image/B11-GD.jpg', 0)
img2 = cv2.resize(img2, (500, 600))

orb = cv2.ORB_create(nfeatures=1000)      # ORB is a fast working algorithm and it's free unlike swift/surf

kp1, des1 = orb.detectAndCompute(img1, None)     # kp1 will be the features, None is meant for the mask
kp2, des2 = orb.detectAndCompute(img2, None)

# descriptors details
# print(des1)
# print(des1.shape)     # (498, 32)
# print(des2.shape)     # (500, 32)

# ORB detector uses 500 features by default. So it will try to find 500 features in the images.
# For each feature, ORB will describe it in 32 values.

# Now that we have the descriptors, we can use a "matcher" to match these descriptors together.
bf = cv2.BFMatcher()        # brute force matcher
matches = bf.knnMatch(des1, des2, k=2)  # k=2 as we want two values that we can compare later.

# To decide whether it's a good match or not
good_matches = []
for m, n in matches:        # m, n are values from k=2
    if m.distance < 0.75 * n.distance:
        good_matches.append([m])

print(len(good_matches))

# Plotting the good matches
img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)

# To view the keypoints
imgkp1 = cv2.drawKeypoints(img1, kp1, None)
imgkp2 = cv2.drawKeypoints(img2, kp2, None)

cv2.imshow("kp1", imgkp1)
cv2.imshow("kp2", imgkp2)

cv2.imshow('img1', img1)
cv2.imshow('img2', img2)
cv2.imshow('img3', img3)

cv2.waitKey(0)