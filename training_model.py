# get other modules from this project
import Img_processing_methodes as met
import utilities as ut
import cv2

# For saving and loading data
import json
import datetime
from pathlib import Path

# training modek
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

import matplotlib.pyplot as plt
import joblib


train_folder = Path("train_data")
train_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist # copilot auto AI
best_train_folder = Path("train_data_best")
best_train_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist # copilot auto AI
model_folder = Path("models")
model_folder.mkdir(exist_ok=True)  # Create the folder if it doesn't exist # copilot auto AI

def train_data(img):
    """
    This function executes all the image processing methodes and returns a list of the results for the training data.
    """
    X_train_lst = []
    X_train_lst.append(met.pistolet_detection(img))
    # X_train_lst.append(met.hough_line_detection(img))
    X_train_lst.extend(met.mean_color(img)) # the return is a list, extend adds to the list instead of adding an list in the list
    return X_train_lst

def to_json(data):
    """
    This function takes a dictionary and saves it as a json file with a timestamp in the name.
    """
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_name = f"train_data_{timestamp}.json"
    train_data_path = train_folder / file_name
    # the idee was my own but the excution was done with the help of AI for this piece of code
    # start
    with open(train_data_path, 'w') as f:
        json.dump(data, f)
    # end
    
def train(use_best_data=False):
    """
    This function makes the Knn model save it for later use and makes a confusion matrix to see how well the model performs. 
    The function returns the X_data and y_data for later use.
    Made with AI help
    """
    # the idee was my own but the excution was done with the help of AI for this piece of code
    # start
    print("starting training...")
    if use_best_data:   # this is so that an good training_data set can always be used
        filename = max(best_train_folder.glob("train_data_*.json"))
    else:
        filename = max(train_folder.glob("train_data_*.json"))
    with open(filename, 'r') as f:
        data = json.load(f)
    y_data, X_data = zip(*data)
    X_data = list(X_data)
    y_data = list(y_data)
    # end
    
    #start AI
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    dectree = RandomForestClassifier()  # https://scikit-learn.org/stable/modules/tree.html
    dectree.fit(X_train, y_train)
    knn = dectree
    # knn = KNeighborsClassifier(n_neighbors=3)
    # knn.fit(X_train, y_train)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")   #own idee. search for the solution with AI
    model_path = model_folder / f"knn_model_{timestamp}.pkl"
    joblib.dump(knn, model_path)  # AI search for joblib things

    y_pred = knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    # Plot confusion matrix
    disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=knn.classes_
)

    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.show()
    print(classification_report(y_test, y_pred))
    # end AI
    return X_data, y_data

def prediction(predict_data): # example: predict_data = [[True, 0.5, 0.5, 0.5]]
    """
    Idee was to save
    Made with the help of AI
    """
    model = joblib.load(max(model_folder.glob("knn_model_*.pkl")))
    prediction = model.predict(predict_data)
    print(f"Prediction: {prediction[0]}")
    

if __name__ == "__main__":
    print("Running train_data.py as main script")
    to_json({"test": [1, 2, 3]})
    
    X, y = train()
    print(f"X: {X}, y: {y}")
