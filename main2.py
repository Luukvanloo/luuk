"""
This code is to apply the training and the prediction of the model
"""

from training_model import train_data, to_json, train, prediction
import cv2
training = True

if training:
    # train()
    train(True)  # Use the best training data for training
else: #prediction
    img = cv2.imread("C:\\Users\\Max Werk\\HU_work\\Beeld\\broodjes\\data\\pistolet_bruin\\frame_000400.jpg")
    img2 = cv2.imread("C:\\Users\\Max Werk\\HU_work\\Beeld\\broodjes\\data\\pistolet_bruin\\frame_000200.jpg")
    data = [train_data(img)]
    print(data)
    prediction(data)
