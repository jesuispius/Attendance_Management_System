# ==================================================================================================================== #
# Creator: Phuoc Nguyen
# Date: March 22, 2021
# Filename: Camera.py
# Description: .
# ==================================================================================================================== #

import tkinter as tk
import Capture as cap
import Train
import Track

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

# ----------------------------------------------------MAIN BUTTONS-----------------------------------------------------#
# Create a button to take the pictures
takeImage_button = tk.Button(window,
                             text="Take Images",
                             command=lambda: cap.takePhotos(id_input.get(), name_input.get(), message_label),
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
                            command=lambda: Train.TrainImages(message_label),
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
                            command=lambda: Track.TrackImages(),
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
