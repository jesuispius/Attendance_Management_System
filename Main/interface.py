import datetime
import time
import tkinter as tk
import csv
import cv2
import numpy as np
import os
import pandas as pd
from PIL import Image

# Create a interface
window = tk.Tk()

# ----------------------------------------------------TITTLE GUI------------------------------------------------------ #
window.title("Face Attendance")

# Change/Update interface/window background color
window.configure(bg='white')

# Set a full-size screen interface/window
# window.attributes('-fullscreen', True)

# ----------------------------------------------------HEADER GUI------------------------------------------------------ #
# Create header of interface
header = tk.Label(window,
                  text="Attendance Management System",
                  bg="yellow",
                  fg="red",
                  width=83,
                  height=3,
                  font=('Georgia', 18, 'bold'))
# Position of header
header.place(x=0, y=0)

# ----------------------------------------------------BODY GUI-------------------------------------------------------- #
# Create an ID label
id_label = tk.Label(window,
                    text="Enter ID:",
                    width=20,
                    fg="red",
                    bg="white",
                    font=('Georgia', 14, ' bold '))
# Position of ID label
id_label.place(x=50, y=120)

# Create an ID input box
id_input = tk.Entry(window,
                    width=20,
                    bg="yellow",
                    fg="red",
                    font=('Georgia', 14, ' bold '))
# Position of ID input box
id_input.place(x=350, y=120)
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# Create an name label
name_label = tk.Label(window,
                      text="Enter Name:",
                      width=20,
                      fg="red",
                      bg="white",
                      font=('Georgia', 14, ' bold '))
# Position of name label
name_label.place(x=50, y=170)

# Create an name input box
name_input = tk.Entry(window,
                      width=20,
                      bg="yellow",
                      fg="red",
                      font=('Georgia', 14, ' bold '))
# Position of name input box
name_input.place(x=350, y=170)
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# Create an noti label
noti_label = tk.Label(window,
                      text="Notification:",
                      width=20,
                      fg="red",
                      bg="white",
                      font=('Georgia', 14, ' bold '))
# Position of noti label
noti_label.place(x=50, y=220)

# Create an message notifications box
message_label = tk.Label(window,
                         text="",
                         width=20,
                         bg="yellow",
                         activebackground="yellow",
                         fg="red",
                         font=('Georgia', 14, ' bold '))
# Position of message notifications box
message_label.place(x=350, y=220)


# ------------------------------------------------------FUNCTIONS----------------------------------------------------- #
# Function to check if input is numerical data or not
def check_integer(input_data):
    """
    Function to check if input is numerical data or not
    :param input_data:
    :return: a boolean result
    """
    try:
        int(input_data)
        return True
    except ValueError:
        pass
    return False


# -------------------------------------------------------------------------------------------------------------------- #
# Function to take the photos
def takePhotos():
    """
        Function to take the photo
        This function will detect the faces and
    """
    # Get data from ID input box
    get_id = (id_input.get())
    # Get data from name input box
    get_name = (name_input.get())

    # Check the validity of id and name
    if check_integer(get_id) and get_name.isalpha():
        # Default webcam
        cam = cv2.VideoCapture(0)
        # Using harcascade for face regconition
        harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sample_num = 1

        while True:
            # Read frame image
            ret, img = cam.read()
            # Resize the image
            img = cv2.resize(img, None, fx=0.8, fy=0.8, interpolation=cv2.INTER_AREA)
            # Convert BGR color to Gray channel
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            # Using built-in function detectMultiScale for face detection
            faces = detector.detectMultiScale(gray, 1.05, 5)

            for (x, y, w, h) in faces:
                # Draw rectangle around face
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 255), 2)
                # Incrementing number of sample
                sample_num = sample_num + 1
                # Saving the captured face in the dataset folder TrainingImageLabel
                cv2.imwrite('C:/Users/PhuocPius/Desktop/PyAttendants/Output/' + str(get_name) + "." +
                            str(get_id) + '.' + str(sample_num) + ".jpg", gray[y:y + h, x:x + w])
                # Display the frame
                cv2.imshow('frame', img)

            # Wait for 100 miliseconds
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # Break if the sample number is morethan 20
            elif sample_num > 20:
                break

        cam.release()
        cv2.destroyAllWindows()

        # Store data in the excel
        res = str(get_id) + " - " + str(get_name) + " - " + "Success!"
        row = [get_id, get_name]
        with open('C:/Users/PhuocPius/Desktop/PyAttendants/StudentDetails/StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message_label.configure(text=res)
    else:
        if check_integer(get_id):
            res = "Enter The Right Name"
            message_label.configure(text=res)
        if get_name.isalpha():
            res = "Enter Right Numerical ID"
            message_label.configure(text=res)


# -------------------------------------------------------------------------------------------------------------------- #
def getImagesAndLabels(path):
    # Get the path of all the files in the folder
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]

    # Empty face list
    faces = []
    # Empty ID list
    IDs = []

    # Looping through all the image paths and loading the IDs and the faces
    for each_path in image_paths:
        # loading the image and converting it to gray scale
        pil_image = Image.open(each_path).convert('L')
        # Now we are converting the PIL image into numpy array
        image_numpy = np.array(pil_image, 'uint8')
        # Getting the Id from the image
        Id = int(os.path.split(each_path)[-1].split(".")[1])
        # Extract the face from the training image sample
        faces.append(image_numpy)
        IDs.append(Id)
    return faces, IDs


# -------------------------------------------------------------------------------------------------------------------- #
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    # Using harcascade for face regconition
    harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels('C:/Users/PhuocPius/Desktop/PyAttendants/Output/')
    recognizer.train(faces, np.array(Id))
    recognizer.save("C:/Users/PhuocPius/Desktop/PyAttendants/TrainingImageLabel/Trainner.yml")
    res = "Image Trained!"
    message_label.configure(text=res)


# -------------------------------------------------------------------------------------------------------------------- #
def TrackImages():
    # Loading recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    # Read data from Trainner.yml
    recognizer.read("C:/Users/PhuocPius/Desktop/PyAttendants/TrainingImageLabel/Trainner.yml")

    # Using hrcascade for face detection
    harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)

    # Read data from StdentDetails.csv
    dataset_file = pd.read_csv("C:/Users/PhuocPius/Desktop/PyAttendants/StudentDetails/StudentDetails.csv")

    # Create a video capture object
    cam = cv2.VideoCapture(0)

    # Select the font and it will be showing the name of person
    font_cv = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)

    while True:
        # Capturing frames from webcam
        ret, img = cam.read()
        # Convert the images to gray scale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect the face in the object
        faces = faceCascade.detectMultiScale(gray, 1.05, 5)

        # Create a rectangle over the face and print the
        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 255), 2)

            # Predict using the confidence
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                # Create a time object
                ts = time.time()
                # Current date
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                # Current timestamp
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                # Check and compare the data from prediction and data from file StudentDetails.csv
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
    fileName = "C:/Users/PhuocPius/Desktop/PyAttendants/Attendance/Attendance_" + date + ".csv"
    attendance.to_csv(fileName, index=False)

    # Destroy and release all windows
    cam.release()
    cv2.destroyAllWindows()


# ----------------------------------------------------MAIN BUTTONS-----------------------------------------------------#
# Create a button to take the pictures
takeImage_button = tk.Button(window,
                             text="Take Images",
                             command=takePhotos,
                             fg="red",
                             bg="yellow",
                             width=20,
                             activebackground="Red",
                             font=('Georgia', 14, ' bold '))
# Position of the take picture button
takeImage_button.place(x=50, y=350)
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# Create a button to train images
trainImg_button = tk.Button(window,
                            text="Train Images",
                            command=TrainImages,
                            fg="red",
                            bg="yellow",
                            width=20,
                            activebackground="Red",
                            font=('Georgia', 14, ' bold '))
# Position of the train-images button
trainImg_button.place(x=350, y=350)
# -------------------------------------------------------------------------------------------------------------------- #
# -------------------------------------------------------------------------------------------------------------------- #
# Create a button to track images
trackImg_button = tk.Button(window,
                            text="Track Images",
                            command=TrackImages,
                            fg="red",
                            bg="yellow",
                            width=20,
                            activebackground="Red",
                            font=('Georgia', 14, ' bold '))
# Position of the train-images button
trackImg_button.place(x=650, y=350)

# ------------------------------------------------------MAINLOOP------------------------------------------------------ #
# Infinite loop used to run the application
window.mainloop()
