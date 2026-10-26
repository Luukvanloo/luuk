# from pathlib import Path
# from utilities import paths_of_single_class_folder, first_image_of_each_object
# import cv2
# import numpy as np

# """
# 0: bastille
# 1: boulogne
# 2: kaiser bruin
# 3: kaiser wit
# 4: pistolet bruin
# 5: pistolet wit
# 6: tijger bol
# """

# # img_paths = paths_of_single_class_folder(3)
# img_paths = first_image_of_each_object()

# for img_path in img_paths:
#      print(img_path.parent.name)

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

# img_paths = paths_of_single_class_folder(3)
img_paths = first_image_of_each_object()

laplacian_imgs = []

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

    if len(contours) == 0:
        continue

    outer_cnt = max(contours, key=cv2.contourArea)

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

    roi = cv2.bitwise_and(
        roi_img,
        roi_img,
        mask=roi_mask
    )

    # ----------------------------
    # Laplacian
    # ----------------------------

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )

    laplacian = cv2.Laplacian(
        gray,
        cv2.CV_16S
    )

    laplacian = cv2.convertScaleAbs(
        laplacian
    )

    # Convert to BGR so colored text can be drawn
    laplacian = cv2.cvtColor(
        laplacian,
        cv2.COLOR_GRAY2BGR
    )

    # Resize all images to same size
    # laplacian = cv2.resize(
    #     laplacian,
    #     (300, 300)
    # )

    # Add label
    cv2.putText(
        laplacian,
        img_path.parent.name,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    laplacian_imgs.append(
        laplacian
    )

# --------------------------------
# Create grid of all 7 images
# --------------------------------

while len(laplacian_imgs) < 8:
    laplacian_imgs.append(
        np.zeros((300, 300, 3),
                 dtype=np.uint8)
    )

top_row = np.hstack(
    laplacian_imgs[:4]
)

bottom_row = np.hstack(
    laplacian_imgs[4:8]
)

grid = np.vstack(
    [top_row, bottom_row]
)

cv2.imshow(
    "All Laplacians",
    grid
)

cv2.waitKey(0)
cv2.destroyAllWindows()