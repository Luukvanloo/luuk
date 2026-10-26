from pathlib import Path
import cv2
import numpy as np

data_dir = Path("data")

class_folders = sorted([
    p for p in data_dir.iterdir()
    if p.is_dir()
])


for class_folder in class_folders:

    class_name = class_folder.name

    img_paths = sorted(
        class_folder.glob("*.jpg")
    )

    for img_path in img_paths:

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        # ----------------------------
        # White background verwijderen
        # ----------------------------

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        lower_white = np.array([0, 0, 50])
        upper_white = np.array([180, 30, 255])

        mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        mask = cv2.bitwise_not(mask)

        # ----------------------------
        # Contours zoeken
        # ----------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            break

        # Grootste contour
        outer_cnt = max(
            contours,
            key=cv2.contourArea
        )

        # Kopie maken voor tekenen
        contour_img = img.copy()

        # Contour groen tekenen
        cv2.drawContours(
            contour_img,
            [outer_cnt],
            -1,
            (0, 255, 0),
            3
        )

        print("Class:", class_name)
        print("Image:", img_path.name)
        print("Aantal contours:", len(contours))

        cv2.imshow(
            f"Contour - {class_name}",
            contour_img
        )

        cv2.imshow(
            "Mask",
            mask
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Alleen eerste foto van deze class
        break
    break

from pathlib import Path
import cv2
import numpy as np

data_dir = Path("data")

class_folders = sorted([
    p for p in data_dir.iterdir()
    if p.is_dir()
])


for class_folder in class_folders:

    class_name = class_folder.name

    img_paths = sorted(
        class_folder.glob("*.jpg")
    )

    for img_path in img_paths:

        img = cv2.imread(str(img_path))

        if img is None:
            continue

        # ----------------------------
        # White background verwijderen
        # ----------------------------

        hsv = cv2.cvtColor(
            img,
            cv2.COLOR_BGR2HSV
        )

        lower_white = np.array([0, 0, 50])
        upper_white = np.array([180, 30, 255])

        mask = cv2.inRange(
            hsv,
            lower_white,
            upper_white
        )

        mask = cv2.bitwise_not(mask)

        # ----------------------------
        # Contours zoeken
        # ----------------------------

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            break

        # Grootste contour
        outer_cnt = max(
            contours,
            key=cv2.contourArea
        )

        # ----------------------------
        # Bounding rectangle
        # ----------------------------

        x, y, w, h = cv2.boundingRect(
            outer_cnt
        )

        bounding_img = img.copy()

        # Groene rectangle tekenen
        cv2.rectangle(
            bounding_img,
            (x, y),
            (x + w, y + h),
            (0, 255, 0),
            3
        )

        print("Class:", class_name)
        print("Image:", img_path.name)

        print(
            "Bounding rect:",
            x,
            y,
            w,
            h
        )

        cv2.imshow(
            f"Bounding Rect - {class_name}",
            bounding_img
        )

        cv2.imshow(
            "Mask",
            mask
        )

        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Alleen eerste foto van deze class
        break
    break