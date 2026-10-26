import Img_processing_methodes as met
import utilities as ut
import cv2 
import training_model as tm


# img_paths = ut.first_image_of_each_object()
img_dict = ut.all_data()
train_data = [] # train_data should look like this: [["pistolet", [0.1, 0.2, 0.3]], ["broodje", [0.4, 0.5, 0.6]]]
#TODO: export all training data in to a database and then use the database to train a model
for img_class, img_paths in img_dict.items():
    for img_path in img_paths:
        print(f"Processing image: {img_path.name} of class: {img_class}")
        img = cv2.imread(str(img_path))
        lines = met.hough_line_detection(img)
        print("Number of lines detected:", lines)
        
        break
        x_train = tm.train_data(img)
        train_data.append([img_class, x_train])
        # met.histogram(img, None)
        # met.canny_edge(img)
        # roi, circles = met.circle_detection(img)
        # met.harris_corner_detection(img)


        #roi = met.crop_biggest_bread(img)
        # if roi is not None:
        #     cv2.imshow("ROI",roi)
        # else:
        #     print(roi)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()


# tm.to_json(train_data)
#TODO: old code for single img detection
# img_paths = ut.first_image_of_each_object()    
# for img_path in img_paths:
    
#         img = cv2.imread(str(img_path))
#         # met.canny_edge(img)
#         roi, circles = met.circle_detection(img)
#         pistolet = met.pistolet_detection(img)
#         print(f"is pistolet: {pistolet}")
#         #roi = met.crop_biggest_bread(img)
#         if roi is not None:
#             cv2.imshow("ROI",roi)
#         else:
#             print(roi)
#         cv2.waitKey(000)
#         cv2.destroyAllWindows()
    
