import img_pre_processing_functions as pre
import utilities as ut
import numpy as np
import cv2
from matplotlib import pyplot as plt
import math

def remove_white_background(img):
    """HSV threshold vwith parameters for removing the background"""
    lower_white = np.array([0, 0, 50])
    upper_white = np.array([180, 30, 255])
    return pre.HSV_threshold(img, lower_value=lower_white, upper_value=upper_white)


def crop_biggest_bread(img):
    """
    Finds the biggest bread piece and crops a roi (rectangle of interest) img.
    """
    mask = remove_white_background(img)
    contours = pre.find_contours(mask)
    print(len(contours))
    # start AI
    if len(contours) == 0:
        # if no contours present
        return None
    biggest_c = max(contours, key=cv2.contourArea)
    object_mask = np.zeros(mask.shape, dtype=np.uint8)
    cv2.drawContours(   # with AI
    object_mask,
    [biggest_c],
    -1,
    255,
    thickness=cv2.FILLED
    )
    # (x,y)-coordinate with of the top right corner of the object abd tge width and height
    #TODO add source for the code beneath
    x, y, w, h = cv2.boundingRect(biggest_c)
    roi_img = img[y:y+h, x:x+w]
    roi_mask = object_mask[y:y+h, x:x+w]
    roi = cv2.bitwise_and(roi_img, roi_img, roi_mask)
    # end AI
    return roi, roi_mask 

def circle_detection(img):
    """
    Hough circles detection
    """
    img = cv2.resize(
            img,
            None,
            fx=0.65,
            fy=0.65,
            interpolation=cv2.INTER_AREA
        )
    # img = cv2.medianBlur(img,5)
    img = remove_white_background(img)
    # img = pre.gaussiann_blur(img, (9,9))
    # img = pre.to_grayscale(img)
    # ---------------------------------------
    # start code citation source https://docs.opencv.org/3.4.5/da/d53/tutorial_py_houghcircles.html
    # ---------------------------------------
    circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,3,70,
                            param1=200,param2=56,minRadius=40,maxRadius=200)
    if circles is None:
        return img, None
    circles = np.uint16(np.around(circles))
    print(len(circles[0]))
    for i in circles[0,:]:
        
        # draw the outer circle
        cv2.circle(img,(i[0],i[1]),i[2],(0,255,0),2)
    # draw the center of the circle
        cv2.circle(img,(i[0],i[1]),2,(0,0,255),3)
    # ---------------------------------------
    # end code citation
    # ---------------------------------------
    
    return img, circles

def pistolet_detection(img):
    """
    An pistolet is an oval. thus by applying
    """ #TODO
    img, circles = circle_detection(img)

    if circles is None:
        return False
    distances = []
    lst_circles = circles[0]
    print(lst_circles)
    length_loop = len(circles[0])-1
    if length_loop > 0:
        for i in range(length_loop):
            x1, y1, r1 = lst_circles[i]
            x2, y2, r2 = lst_circles[i+1]
            x1, y1 = int(x1), int(y1)   # Overflow error solved by AI by copy pasting the error to copilot
            x2, y2 = int(x2), int(y2)
            dis = math.sqrt((x2-x1)**2 + (y2-y1)**2) # pytharos. 
            print(f"distances from middelpoints: {dis}")
            distances.append(dis - r1 - r2) # if the distance minus the 2 radius. Then the circles overlap
    # This was to get make use of all the distances. Altough it might be better to do it in a way that if 1 is negative then its
    # not a pistolet
    if len(distances) > 0:
        avg_dis = sum(distances)/len(distances)
        print(f"all distances: {distances}")
        print(f"avg_dis:{avg_dis}")
        return True if avg_dis > 10 else False
    else:
        return False

def hough_line_detection(img):
    """
    Detect and draw Hough lines.
    made by visual studio code AI
    """

    # Resize
    img = cv2.resize(
        img,
        None,
        fx=0.65,
        fy=0.65,
        interpolation=cv2.INTER_AREA
    )

    # Convert to grayscale
    gray = pre.to_grayscale(img)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    print(edges)
    print(f"edges shape: {edges.shape}, edges dtype: {edges.dtype}")
    # Detect lines
    lines = cv2.HoughLines(edges, 1, np.pi / 180, 50)

    # Create color image for drawing
    draw_img = pre.to_grayscale(img)

    if lines is None:
        return 0
    else:
        return len(lines)

    # Draw all detected lines
    # for line in lines:
    #     rho, theta = line[0]

    #     a = np.cos(theta)
    #     b = np.sin(theta)

    #     x0 = a * rho
    #     y0 = b * rho

    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * a)

    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * a)

    #     cv2.line(draw_img, (x1, y1), (x2, y2), (0, 0, 255), 2)

    # return draw_img, lines


def show_hough_lines(img):
    """
    
    Show the detected Hough lines on the image.
    Made by AI"""
    img_lines, lines = hough_line_detection(img)

    print(f"Detected {0 if lines is None else len(lines)} lines")

    plt.imshow(cv2.cvtColor(img_lines, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()


def canny_edge(img):
    """
    Perform canny edge algorithm and show the results in an img.
    Made using https://docs.opencv.org/4.1.1/da/d22/tutorial_py_canny.html
    """
    img = remove_white_background(img)
    edges = cv2.Canny(img,100,200)
    print(edges)
    plt.subplot(121),plt.imshow(img,cmap = 'gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122),plt.imshow(edges,cmap = 'gray')
    plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
    plt.show()

def histogram(img, mask):
    """
    Calculates the histogram of an image with a mask.
    made by visual studio code AI
    """
    mask = remove_white_background(img)
    hist = cv2.calcHist([img],[0],mask,[256],[0,256])
    plt.plot(hist)
    plt.xlim([0,256])
    plt.show()

def harris_corner_detection(img):
    """
    Detects corners in an image using the Harris corner detection method.
    AI code source: https://docs.opencv.org/4.x/dc/d0d/tutorial_py_features_harris.html
    Made visual studio code AI
    """
    img = cv2.resize(
        img,
        None,
        fx=0.65,
        fy=0.65,
        interpolation=cv2.INTER_AREA
    )
    gray = pre.to_grayscale(img)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, 2, 7, 0.04)

    # Result is dilated for marking the corners
    dst = cv2.dilate(dst, None)

    # Threshold for an optimal value, it may vary depending on the image.
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def orb_feature_detection(img):
    """
    Detects features in an image using the ORB (Oriented FAST and Rotated BRIEF) method.
    Made by visual studio code AI
    """
    img = cv2.resize(
        img,
        None,
        fx=0.65,
        fy=0.65,
        interpolation=cv2.INTER_AREA
    )
    gray = pre.to_grayscale(img)
    blur = cv2.GaussianBlur(gray, (0,0), 3)
    gray = cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    img_with_keypoints = cv2.drawKeypoints(img, keypoints, None, color=(0, 255, 0), flags=0)

    plt.imshow(cv2.cvtColor(img_with_keypoints, cv2.COLOR_BGR2RGB))
    plt.axis("off")
    plt.show()

def extract_mean_color(img):
    """
    This function extract the mean color of the img on the RGB spectrum
    in order B, G, R
    https://docs.opencv.org/4.13.0/d1/d32/tutorial_py_contour_properties.html#autotoc_md1343
    """
    mean_color = cv2.mean(img)[:3]  # Get the mean color in BGR format
    return mean_color

def mean_color(img):
    """
    This function extract the mean color of the img on the RGB spectrum
    in order B, G, R
    https://docs.opencv.org/4.13.0/d1/d32/tutorial_py_contour_properties.html#autotoc_md1343
    """ 
    roi_img, roi_mask = crop_biggest_bread(img)
    return cv2.mean(
                roi_img,
                mask=roi_mask
            )[:3] 
