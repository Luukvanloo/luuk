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

for img_path in img_paths:

    print(img_path.parent.name)

    img = cv2.imread(str(img_path))
    
    if img is None:
        continue

    img = cv2.resize(
        img,
        None,
        fx=0.4,
        fy=0.4,
        interpolation=cv2.INTER_AREA
    )
    #------------
    # Test img
    #------------
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #------------
    # Separate RGB
    #------------
    # b, g, r = cv2.split(img)

    # zeros = np.zeros_like(b)

    # # Colored versions
    # r_img = cv2.merge([zeros, zeros, r])
    # g_img = cv2.merge([zeros, g, zeros])
    # b_img = cv2.merge([b, zeros, zeros])

    # # Original + grayscale
    # cv2.imshow("RGB", img)
    # cv2.imshow("Grayscale", gray1)

    # # Grayscale RGB channels
    # cv2.imshow("Red - Grayscale", r)
    # cv2.imshow("Green - Grayscale", g)
    # cv2.imshow("Blue - Grayscale", b)

    # # Colored RGB channels
    # cv2.imshow("Red - Color", r_img)
    # cv2.imshow("Green - Color", g_img)
    # cv2.imshow("Blue - Color", b_img)

    # cv2.waitKey(0)
        #------------
    # Separate HSV
    #------------
        #------------
    # Separate HSV
    #------------
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)


    lower_white = np.array([0, 0, 50])
    upper_white = np.array([180, 30, 255])

    mask = cv2.inRange(hsv, lower_white, upper_white)
    mask = cv2.bitwise_not(mask)
    # h, s, v = cv2.split(hsv)

    # # Colored Hue
    # h_color = cv2.merge([
    #     h,
    #     np.full_like(h, 255),
    #     np.full_like(h, 255)
    # ])
    # h_color = cv2.cvtColor(h_color, cv2.COLOR_HSV2BGR)

    # # Colored Saturation
    # s_color = cv2.merge([
    #     h,
    #     s,
    #     np.full_like(v, 255)
    # ])
    # s_color = cv2.cvtColor(s_color, cv2.COLOR_HSV2BGR)

    # # Colored Value
    # v_color = cv2.merge([
    #     h,
    #     np.full_like(s, 255),
    #     v
    # ])
    # v_color = cv2.cvtColor(v_color, cv2.COLOR_HSV2BGR)

    # # Original
    # cv2.imshow("RGB", img)

    # # Grayscale HSV channels
    # cv2.imshow("Hue - Grayscale", h)
    # cv2.imshow("Saturation - Grayscale", s)
    # cv2.imshow("Value - Grayscale", v)

    # # Colored HSV channels
    # cv2.imshow("Hue - Color", h_color)
    # cv2.imshow("Saturation - Color", s_color)
    # cv2.imshow("Value - Color", v_color)

    # cv2.waitKey(0)
    #------------
    # Test end
    #------------
    #------------
    # Binary threshold
    #------------
    # threshold_value = 127

    # _, binary_img = cv2.threshold(
    #     gray1,
    #     threshold_value,
    #     255,
    #     cv2.THRESH_BINARY
    # )

    # cv2.imshow("Grayscale", gray1)
    # cv2.imshow("Binary", binary_img)

    # cv2.waitKey(0)
    #     #------------
    # # Truncated threshold
    # #------------
    # threshold_value = 127

    # _, truncated_img = cv2.threshold(
    #     gray1,
    #     threshold_value,
    #     255,
    #     cv2.THRESH_TRUNC
    # )

    # cv2.imshow("Grayscale", gray1)
    # cv2.imshow("Truncated", truncated_img)

    # cv2.waitKey(0)
        #------------
    # To-zero threshold
    #------------
    # threshold_value = 127

    # _, tozero_img = cv2.threshold(
    #     gray1,
    #     threshold_value,
    #     255,
    #     cv2.THRESH_TOZERO
    # )

    # cv2.imshow("Grayscale", gray1)
    # cv2.imshow("To-zero", tozero_img)

    # cv2.waitKey(0)
        #------------
    # Adaptive threshold
    #------------
    # adaptive_img = cv2.adaptiveThreshold(
    #     gray1,
    #     255,
    #     cv2.ADAPTIVE_THRESH_MEAN_C,
    #     cv2.THRESH_BINARY,
    #     11,
    #     2
    # )

    # cv2.imshow("Grayscale", gray1)
    # cv2.imshow("Adaptive Threshold", adaptive_img)

    # cv2.waitKey(0)
        #------------
    # Otsu's Binarization
    #------------
    # otsu_value, otsu_img = cv2.threshold(
    #     gray1,
    #     0,
    #     255,
    #     cv2.THRESH_BINARY + cv2.THRESH_OTSU
    # )

    # print("Otsu threshold:", otsu_value)

    # cv2.imshow("Grayscale", gray1)
    # cv2.imshow("Otsu", otsu_img)

    # cv2.waitKey(0)

    #------------
    # 2D Convolution
    #------------
    # kernel = np.ones((5, 5), np.float32) / 25

    # convolution_img = cv2.filter2D(
    #     img,
    #     -1,
    #     kernel
    # )

    # cv2.imshow("Original", img)
    # cv2.imshow("2D Convolution", convolution_img)

    # cv2.waitKey(0)
    #------------
    # Averaging
    #------------
    # average_img = cv2.blur(
    #     img,
    #     (5, 5)
    # )

    # cv2.imshow("Original", img)
    # cv2.imshow("Averaging", average_img)

    # cv2.waitKey(0)
    #------------
    # Gaussian Blurring
    #------------
    # gaussian_img = cv2.GaussianBlur(
    #     img,
    #     (5, 5),
    #     0
    # )

    # cv2.imshow("Original", img)
    # cv2.imshow("Gaussian Blur", gaussian_img)

    # cv2.waitKey(0)
        #------------
    # Median Blurring
    #------------
    # median_img = cv2.medianBlur(
    #     img,
    #     5
    # )

    # cv2.imshow("Original", img)
    # cv2.imshow("Median Blur", median_img)

    # cv2.waitKey(0)
    #------------
    # Bilateral Filtering
    #------------
    # bilateral_img = cv2.bilateralFilter(
    #     img,
    #     9,
    #     75,
    #     75
    # )

    # cv2.imshow("Original", img)
    # cv2.imshow("Bilateral Filter", bilateral_img)

    # cv2.waitKey(0)
        #------------
    # Morphological Transformations
    #------------

     #------------
    # Morphological Transformations
    #------------

    # Convert to grayscale
    gray_morph = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Convert to binary AND invert colors
    # Object = white, background = black
    _, binary_morph = cv2.threshold(
        gray_morph,
        127,
        255,
        cv2.THRESH_BINARY_INV
    )

    # Create 5x5 kernel
    kernel = np.ones((5, 5), np.uint8)


    #------------
    # 1. Erosion
    #------------
    erosion_img = cv2.erode(
        binary_morph,
        kernel,
        iterations=1
    )


    #------------
    # 2. Dilation
    #------------
    dilation_img = cv2.dilate(
        binary_morph,
        kernel,
        iterations=1
    )


    #------------
    # 3. Opening
    #------------
    opening_img = cv2.morphologyEx(
        binary_morph,
        cv2.MORPH_OPEN,
        kernel
    )


    #------------
    # 4. Closing
    #------------
    closing_img = cv2.morphologyEx(
        binary_morph,
        cv2.MORPH_CLOSE,
        kernel
    )


    #------------
    # 5. Morphological Gradient
    #------------
    gradient_img = cv2.morphologyEx(
        binary_morph,
        cv2.MORPH_GRADIENT,
        kernel
    )


    #------------
    # 6. Top Hat
    #------------
    tophat_img = cv2.morphologyEx(
        binary_morph,
        cv2.MORPH_TOPHAT,
        kernel
    )


    #------------
    # 7. Black Hat
    #------------
    blackhat_img = cv2.morphologyEx(
        binary_morph,
        cv2.MORPH_BLACKHAT,
        kernel
    )


    #------------
    # Show all images
    #------------
    cv2.imshow("Base Image", img)
    cv2.imshow("Binary Inverted", binary_morph)

    cv2.imshow("1 - Erosion", erosion_img)
    cv2.imshow("2 - Dilation", dilation_img)
    cv2.imshow("3 - Opening", opening_img)
    cv2.imshow("4 - Closing", closing_img)
    cv2.imshow("5 - Morphological Gradient", gradient_img)
    cv2.imshow("6 - Top Hat", tophat_img)
    cv2.imshow("7 - Black Hat", blackhat_img)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


    cv2.imshow("base",img)
    cv2.imshow("HSV", hsv)
    cv2.imshow("Threshhold", mask)
    cv2.waitKey(000)
    cv2.destroyAllWindows()
    break

# OUTSIDE the loop
cv2.waitKey(0)
cv2.destroyAllWindows()    
    
cv2.waitKey(0)
