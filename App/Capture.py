# ==================================================================================================================== #
# Creator: Phuoc Nguyen
# Date: March 22, 2021
# Filename: Camera.py
# Description: .
# ==================================================================================================================== #

import csv
import cv2
import Tool as o_tl
from Camera import Camera

# Using harcascade for face regconition
harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(harcascadePath)


# Function to take the photos
def takePhotos(ID, name, message):
    """
        Function to take the photo
        This function will detect the faces and
    """
    # Check the validity of id and name
    if o_tl.checkInteger(ID) and name.isalpha():
        # Default webcam
        cam = Camera(0)
        count = 1

        while True:
            # Prepare camera and get frames (images)
            img, gray = cam.getFrames()

            # Using built-in function detectMultiScale for face detection
            faces = detector.detectMultiScale(gray, 1.05, 5)

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 255), 2)

                # Incrementing number of sample
                count = count + 1

                # Saving the captured face in the dataset folder RawCapturedPicture
                cv2.imwrite('../RawCapturedPicture/' + str(name) + "_" +
                            str(ID) + '_' + str(count) + ".jpg", gray[y:y + h, x:x + w])

                # Display the frame
                cv2.imshow('Capture', img)

            # Wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # Break if the number of frames is more than 100
            elif count > 100:
                break

        # Release and destroy all windows
        cam.releaseCam()

        # Store data in the excel
        res = str(ID) + " - " + str(name) + " - " + "Success!"
        row = [ID, name]
        with open('../Students/StudentsList.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if o_tl.checkInteger(ID):
            res = "Enter The Right Name"
            message.configure(text=res)
        if name.isalpha():
            res = "Enter Right Numerical ID"
            message.configure(text=res)
