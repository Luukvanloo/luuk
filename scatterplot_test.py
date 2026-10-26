from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt


data_dir = Path("data")

# Hier komen ALLE metingen tijdelijk in te staan
# Structuur:
# [class_name, R, G, B]
colour_data = []


# ============================================================
# LOOP DOOR ALLE CLASS FOLDERS
# ============================================================

class_folders = sorted([
    p for p in data_dir.iterdir()
    if p.is_dir()
])


for class_folder in class_folders:

    # Bijvoorbeeld:
    # data/object_0  ->  class_name = "object_0"
    class_name = class_folder.name

    img_paths = sorted(class_folder.glob("*.jpg"))


    # ========================================================
    # LOOP DOOR ALLE FOTO'S VAN DEZE CLASS
    # ========================================================

    for img_path in img_paths:

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
        # ====================================================
        # REMOVE WHITE BACKGROUND
        # ====================================================

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_white = np.array([0, 0, 50])
        upper_white = np.array([180, 30, 255])

        mask = cv2.inRange(hsv, lower_white, upper_white)
        mask = cv2.bitwise_not(mask)


        # Cleanup mask
        kernel = np.ones((5, 5),np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if len(contours) == 0:
            continue

        outer_cnt = max(
            contours,
            key=cv2.contourArea
        )
        # Begin AI
        object_mask = np.zeros(
            mask.shape,
            dtype=np.uint8
        )
        cv2.drawContours(
            object_mask,
            [outer_cnt],
            -1,
            255,
            -1
        )
        # end AI

        # ====================================================
        # CROP ROI
        # ====================================================

        x, y, w, h = cv2.boundingRect(outer_cnt)

        roi_img = img[
            y:y+h,
            x:x+w
        ]

        roi_mask = object_mask[
            y:y+h,
            x:x+w
        ]


        # ====================================================
        # MEAN COLOUR VAN ROI
        # ====================================================

        # OpenCV geeft BGR terug
        mean_b, mean_g, mean_r = cv2.mean(
            roi_img,
            mask=roi_mask
        )[:3]


        # ====================================================
        # DATA IN LIST OPSLAAN
        # ====================================================

        # Gewenste structuur:
        # [object_x, R, G, B]

        colour_data.append([
            class_name,
            mean_r,
            mean_g,
            mean_b
        ])


# ============================================================
# PRINT DATA
# ============================================================

# print("\nColour data:")

# for data in colour_data:
#     print(data)
print("start scatterplot")

# ============================================================
# 3D SCATTERPLOT
# ============================================================

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection="3d")


# Alle unieke classes vinden
classes = sorted(set(
    data[0] for data in colour_data
))


# Elke class afzonderlijk plotten
for class_name in classes:

    class_data = [
        data for data in colour_data
        if data[0] == class_name
    ]

    # colour_data structuur:
    # [class_name, R, G, B]

    R = [data[1] for data in class_data]
    G = [data[2] for data in class_data]
    B = [data[3] for data in class_data]


    ax.scatter(
        R,
        G,
        B,
        label=class_name,
        s=40,
        alpha=0.7
    )


# ============================================================
# GRAPH SETTINGS
# ============================================================

ax.set_xlabel("Red")
ax.set_ylabel("Green")
ax.set_zlabel("Blue")

# ============================================================
# AUTOMATISCH INZOOMEN OP DE DATA
# ============================================================

all_R = np.array([data[1] for data in colour_data])
all_G = np.array([data[2] for data in colour_data])
all_B = np.array([data[3] for data in colour_data])

margin = 5

ax.set_xlim(
    max(0, all_R.min() - margin),
    min(255, all_R.max() + margin)
)

ax.set_ylim(
    max(0, all_G.min() - margin),
    min(255, all_G.max() + margin)
)

ax.set_zlim(
    max(0, all_B.min() - margin),
    min(255, all_B.max() + margin)
)

ax.set_title("RGB colour clouds per class")

ax.legend()

plt.show()
# ===================