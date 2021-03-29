# ==================================================================================================================== #
# Creator: Phuoc Nguyen
# Date: March 22, 2021
# Filename: ModelProcessor.py
# Description:
# ==================================================================================================================== #

import os
import numpy as np
from PIL import Image
from App import Tool as o_tl


def imageFileProcessor(path):
    """
    Convert the PIL image to numpy array, extract the faces and ids from the trained image sample
    :param path:
    :return: faces, and ids
    """
    # Show all files in RawCapturedPicture
    # ..., and get the completed path files
    img_paths = []
    for ea in o_tl.showAllFiles(path):
        img_paths.append(os.path.join(path, ea))

    # Empty face list
    faces = []
    # Empty ID list
    IDs = []

    # Looping through all the image paths and loading the IDs and the faces
    for each_path in img_paths:
        # Loading the image and converting it to gray scale
        pil_img = Image.open(each_path).convert('L')
        # Converting the PIL image into numpy array
        image_numpy = np.array(pil_img, 'uint8')
        # Getting the Id from the image
        Id = int(os.path.split(each_path)[-1].split("_")[1])
        # Extract the face from the training image sample
        faces.append(image_numpy)
        IDs.append(Id)
    return faces, IDs
