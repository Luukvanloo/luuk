# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 16:18:25 2026

@author: maxva
"""
from pathlib import Path
import cv2




def paths_of_single_class_folder(folder_number):
    """
    Creates a path list of all images in a 'folder_number' paths
    with AI help
    """
    
    data_dir = Path("data")
    
    object_folders = sorted([p for p in data_dir.iterdir() if p.is_dir()])
    test_object = object_folders[folder_number]
    return sorted(test_object.glob("*.jpg"))


def first_image_of_each_object():
    """
    Returns a list containing the first .jpg image from each object folder.
    With AI help
    """
    data_dir = Path("data")
    object_folders = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    first_images = []

    for folder in object_folders:
        images = sorted(folder.glob("*.jpg"))
        if images:  # folder is not empty
            first_images.append(images[0])

    return first_images

def all_data():
    """
    extracts all data inside of a directory. with as key the type of bread and as value the list of all img_paths of this bread.
    """
    data_dir = Path("data")
    dir_data = {}
    
    data_folders = sorted([p for p in data_dir.iterdir() if p.is_dir()]) # made by AI
    for data_folder in data_folders:
        dir_data[data_folder.name] = sorted(data_folder.glob("*.jpg")) # sorted() part is made by AI from first_image_of_each_object()
    return dir_data


def extract_image(path_image):
    img = cv2.imread(str(path_image))
    
    if img is None:
        """
        in loop 
        if img == None:
            continue
        """
        return None
    
    img = cv2.resize(
            img,
            None,
            fx=0.65,
            fy=0.65,
            interpolation=cv2.INTER_AREA
        )
    return img

if __name__ == "__main__":
    img_paths = all_data()
    print(img_paths)