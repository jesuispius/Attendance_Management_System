# ==================================================================================================================== #
# Creator: Phuoc Nguyen
# Date: March 22, 2021
# Filename: Train.py
# Description: .
# ==================================================================================================================== #

import cv2
import numpy as np
import ModelProcessor
import Tool as o_tl


def TrainImages(notifications):
    """
    Function to train the model, then save the model to the Trainer.yml
    Lastly, it automatically removes all raw images files to reduce the memory capacity.
    :param notifications:
    :return: None
    """
    recognizer = cv2.face_LBPHFaceRecognizer.create()

    # Get the images from folder, that stores the raw images
    faces, Id = ModelProcessor.imageFileProcessor('../RawCapturedPicture/')

    # Process and save the model to the Trainer.yml
    recognizer.train(faces, np.array(Id))
    recognizer.save("../ModelTrainer/Trainer.yml")

    # Show the message to confirm whether model is trained or not!
    res = "Image Trained!"
    notifications.configure(text=res)

    # After training model
    # ..., delete all captured photos
    o_tl.deleteContentsDir('../RawCapturedPicture')
