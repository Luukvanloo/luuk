from pathlib import Path
from utilities import paths_of_single_class_folder, first_image_of_each_object
import cv2
import numpy as np

"""
0: bastille
1: boulogne
2: kaiser bruin
3: kaiser wit
4: pistolet bruin
5: pistolet wit
6: tijger bol
"""

img_paths = paths_of_single_class_folder(0)
# img_paths = first_image_of_each_object()

for img_path in img_paths:

    print(img_path.parent.name)

    img = cv2.imread(str(img_path))

    if img is None:
        continue

    img = cv2.resize(
        img,
        None,
        fx=0.65,
        fy=0.65,
        interpolation=cv2.INTER_AREA
    )

    # ----------------------------
    # Remove white background
    # ----------------------------

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 50])
    upper_white = np.array([180, 30, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_not(mask)

    # Cleanup mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # ----------------------------
    # Find outer object
    # ----------------------------

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    print(contours)
    if len(contours) == 0:
        continue

    # Largest contour = object
    outer_cnt = max(contours, key=cv2.contourArea)

    # Draw filled object mask
    object_mask = np.zeros(mask.shape, dtype=np.uint8)

    cv2.drawContours(
        object_mask,
        [outer_cnt],
        -1,
        255,
        -1
    )

    # ----------------------------
    # Crop object
    # ----------------------------

    x, y, w, h = cv2.boundingRect(outer_cnt)

    roi_img = img[y:y+h, x:x+w]
    roi_mask = object_mask[y:y+h, x:x+w]

    # Keep only object pixels
    roi = cv2.bitwise_and(
        roi_img,
        roi_img,
        roi_mask
    )
    # ----------------------------
    # test colour values on rgb scale
    # ----------------------------
    mean_color = cv2.mean(roi)[:3]
    median = np.median(roi.reshape(-1,3), axis=0)
    # print(mean_color)
    print(median)
    # ----------------------------
    # Find inner shape
    # ----------------------------

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    laplacian = cv2.Laplacian(gray, cv2.CV_16U)
    laplacian = np.uint8(np.absolute(laplacian))

    _, thresh = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Remove border left by mask
    thresh = cv2.bitwise_and(
        thresh,
        thresh,
        mask=roi_mask
    )

    inner_contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,
    minDist=50
    )




    
    # if len(inner_contours) > 0:

    #     inner_cnt = max(
    #         inner_contours,
    #         key=cv2.contourArea
    #     )

    #     peri = cv2.arcLength(
    #         inner_cnt,
    #         True
    #     )

    #     approx = cv2.approxPolyDP(
    #         inner_cnt,
    #         0.02 * peri,
    #         True
    #     )

    #     vertices = len(approx)

    #     print("Inner vertices:", vertices)

    #     cv2.drawContours(
    #         roi,
    #         [inner_cnt],
    #         -1,
    #         (0, 0, 255),
    #         2
    #     )

    # ----------------------------
    # Display
    # ----------------------------

    cv2.imshow("Mask", mask)
    cv2.imshow("ROI", roi)
    # cv2.imshow("Laplacian", laplacian)
    # cv2.imshow("Threshold", thresh)

    # cv2.waitKey(3000)
    # cv2.waitKey(000)
    # cv2.destroyAllWindows()

    # HSV greyscaled img
    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # h = hsv[:,:,0]
    # s = hsv[:,:,1]
    # v = hsv[:,:,2]

    # cv2.imshow("H", h)
    # cv2.imshow("S", s)
    # cv2.imshow("V", v)

    
    # Create SIFT detector


    # sift = cv2.SIFT_create()

    # kp = sift.detect(gray, None)

    # sift_img = cv2.drawKeypoints(
    #     roi,
    #     kp,
    #     None,
    #     flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    # )

    # cv2.imshow(
    #     f"SIFT - {img_path.name}",
    #     sift_img
    # )
    cv2.waitKey(000)
    
# OUTSIDE the loop
cv2.waitKey(0)
cv2.destroyAllWindows()    
    
cv2.waitKey(0)
cv2.destroyAllWindows()
