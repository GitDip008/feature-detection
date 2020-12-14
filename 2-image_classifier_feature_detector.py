import cv2
import numpy as np
import os

path = 'Color_Image'
images = []
product_names = []
product_list = os.listdir(path)
print(product_list)
print("Total products detected:", len(product_list))

for product in product_list:
    img_current = cv2.imread(f'{path}/{product}', 0)    # 0 for GrayScale
    images.append(img_current)
    product_names.append(os.path.splitext(product)[0])
print(product_names)

# Now we need to find the descriptors of all the given images
orb = cv2.ORB_create(nfeatures=1000)      # ORB is a fast working algorithm and it's free unlike swift/surf

def find_descriptor(images):
    descriptor_list = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        descriptor_list.append(des)
    return descriptor_list


# A function to detect the product id of the most matching descriptor.
def find_id(img, descriptpor_list, threshold=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matches_list = []
    final_value = -1
    try:
        for des in descriptor_list:
            matches = bf.knnMatch(des, des2, k=2)
            # To decide whether it's a good match or not
            good_matches = []
            for m, n in matches:        # m, n are values from k=2
                if m.distance < 0.75 * n.distance:
                    good_matches.append([m])
            matches_list.append(len(good_matches))
        # print(matches_list)
    except:
        pass

    if len(matches_list) != 0:
        if max(matches_list) > threshold:
            final_value = matches_list.index(max(matches_list))

    return final_value


descriptor_list = find_descriptor(images)
print(len(descriptor_list))

# Capturing images from live video
cap = cv2.VideoCapture(0)
while True:
    success, img2 = cap.read()
    img_original = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    id = find_id(img2, descriptor_list)
    if id != -1:
        cv2.putText(img_original, product_names[id], (50, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255), 2)

    cv2.imshow("img2", img_original)
    cv2.waitKey(1)

