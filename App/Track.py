# ==================================================================================================================== #
# Creator: Phuoc Nguyen
# Date: March 22, 2021
# Filename: Track.py
# Description: .
# ==================================================================================================================== #

import datetime
import time
import cv2
import pandas as pd
from Camera import Camera

# Using harcascade for face regconition
harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(harcascadePath)


def TrackImages():
    """
    Track the face objects through the camera
    If system cannot recognize the actual or trained faces, it will show the "Unknown".
    Otherwise, it will be showing the real name and ID over the actual face detected.
    :return: None
    """
    # Loading recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Read data from Trainer.yml
    recognizer.read("../ModelTrainer/Trainer.yml")

    # Read data from StdentDetails.csv
    dataset_file = pd.read_csv("../Students/StudentsList.csv")

    # Create a video capture object
    cam = Camera(0)

    # Select the font and it will be showing the name of person
    font_cv = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        # Prepare camera and get frames (images)
        img, gray = cam.getFrames()

        # Detect the face in the object
        faces = detector.detectMultiScale(gray, 1.05, 5)

        # Create a rectangle over the face and print the
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 255), 2)

            # Predict using the confidence
            Id, confident = recognizer.predict(gray[y:y + h, x:x + w])
            if confident < 50:
                # Create a time object
                ts = time.time()
                # Current date
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                # Current timestamp
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                # Check and compare the data from prediction and data from file Students.csv
                # Return the name that is equivalent to the accurate ID.
                aa = dataset_file.loc[dataset_file['Id'] == Id]['Name'].values
                # Create a string of information, used to add in the face object detected later on!
                tt = str(Id) + "-" + aa
                # Save data to the attendace
                attendance.loc[len(attendance)] = [Id, aa, date, timeStamp]
            else:
                # In case the system cannot recognize the object
                # It will be showing the "Unknown" (name) over the face object detected
                Id = 'Unknown'
                tt = str(Id)

            # Add the name over the face object detected
            cv2.putText(img, str(tt), (x, y + h), font_cv, 1, (0, 128, 255), 2)

        # Remove the duplicate rows based on all columns
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')

        # Show the window
        cv2.imshow('Image', img)

        # Wait until to press 'q' from keyboard to terminate the window
        if cv2.waitKey(1) == ord('q'):
            break

    # Create a time object
    # Purpose: using the current date as a filename for attendance checking.
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    # Create a new filename and save the attendance to that file.
    fileName = "../Attendance/Attendance_" + date + ".csv"
    attendance.to_csv(fileName, index=False)

    # Destroy and release all windows
    cam.releaseCam()
