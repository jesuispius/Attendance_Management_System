# ==================================================================================================================== #
# Creator: Phuoc Nguyen
# Date: March 22, 2021
# Filename: Camera.py
# Description: .
# ==================================================================================================================== #

import cv2


class Camera:
    """
    This class to capture videos, images, depended on IP cam or webcam, etc.

    Construction:
            Ex: Camera(0)       : For webcam

    Functions:
            + releaseCam()        : Release and destroy all windows
            + getFrames()         : Read the frames (images) from camera, return the image and its gray color version
    """
    def __init__(self, IPCam):
        """
        Constructor
        :param IPCam:
                + Ex: For webcam, using IPCam = 0
        """
        self.cam = cv2.VideoCapture(IPCam)

    def releaseCam(self):
        """
        Release and destroy all windows
        :return:
        """
        self.cam.release()
        cv2.destroyAllWindows()

    def getFrames(self):
        """
        Read frames (images) from camera
        :return: the image and its gray color version
        """
        # Read frame image
        ret, img = self.cam.read()
        # Resize the image
        img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
        # Convert BGR color to Gray channel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img, gray
