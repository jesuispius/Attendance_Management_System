import cv2
import pandas as pd
import numpy as np
from App import ModelProcessor
from App import Tool

# Kivy Modules
from kivy.app import App
from kivy.lang import Builder
from kivy.uix.screenmanager import ScreenManager, Screen
from kivy.properties import ObjectProperty
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.popup import Popup
from kivy.uix.label import Label

# Using harcascade for face regconition
harcascadePath = cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
detector = cv2.CascadeClassifier(harcascadePath)

# Loading recognizer
recognizer_track = cv2.face.LBPHFaceRecognizer_create()
# Read data from Trainer.yml
recognizer_track.read("../ModelTrainer/Trainer.yml")

# Read data from StdentDetails.csv
dataset_file = pd.read_csv("../Students/StudentsList.csv")
font_cv = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================================================ #
class TrackFace(Screen, Image):
    def __init__(self, **kwargs):
        super(TrackFace, self).__init__(**kwargs)
        self.capture = CamGenerator(IPCam=0)

    def Track(self):
        if not self.capture.isOpened():
            self.capture = CamGenerator(IPCam=0)
        Clock.schedule_interval(self.Update, 1.0 / 35)

    def DestroyTracking(self):
        self.capture.stop()

    def Update(self, dt):
        ret, img = self.capture.read()
        if ret:
            # Dimension of original image
            rows = img.shape[0]
            cols = img.shape[1]

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.05, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 81, 0), 2)

                # Predict using the confidence
                Id, confident = recognizer_track.predict(gray[y:y + h, x:x + w])
                if confident < 50:
                    aa = dataset_file.loc[dataset_file['Id'] == Id]['Name'].values
                    # Create a string of information, used to add in the face object detected later on!
                    tt = str(Id) + "-" + aa
                else:
                    # In case the system cannot recognize the object
                    # It will be showing the "Unknown" (name) over the face object detected
                    Id = 'Unknown'
                    tt = str(Id)

                # Add the name over the face object detected
                cv2.putText(img, str(tt), (x, y + h), font_cv, 1, (255, 81, 0), 1, cv2.LINE_AA, False)

            img_rotated = cv2.flip(img, 0)
            buf = img_rotated.tobytes()
            image_texture = Texture.create(size=(cols, rows), colorfmt='bgr', bufferfmt='ubyte')
            image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
            # Display image from the texture
            self.texture = image_texture


# ============================================================================================ #
class CamGenerator:
    def __init__(self, IPCam=0, **kwargs):
        super().__init__(**kwargs)
        self.IPCam = IPCam
        self.title = "Attendance Management System"
        self.capture = cv2.VideoCapture(IPCam)

    def isOpened(self):
        if self.capture.isOpened():
            return True
        return False

    def read(self):
        return self.capture.read()

    def stop(self):
        self.capture.release()


# ============================================================================================ #
class CaptureFormWindow(Screen):
    name_person = ObjectProperty(None)
    id_person = ObjectProperty(None)
    ip_cam = ObjectProperty(None)

    def submit(self):
        if self.name_person.text != "" and self.id_person.text != "":
            sm.current = "capture_window"

    def reset(self):
        self.ip_cam.text = ""
        self.id_person.text = ""
        self.name_person.text = ""


# ============================================================================================ #
class CaptureWindow(Screen, Image):
    def __init__(self, **kwargs):
        super(CaptureWindow, self).__init__(**kwargs)
        self.capture = CamGenerator(IPCam=0)
        self.clock_interval = None
        self.get_name_person = None
        self.get_id_person = None
        self.get_ip_cam = None
        self.count_frames = 0

    def TakePhoto(self):
        self.getInfoCaptureForm()
        if not self.capture.isOpened():
            self.capture = CamGenerator(IPCam=self.get_ip_cam)
        self.clock_interval = Clock.schedule_interval(self.Update, 1.0 / 35)
        Clock.schedule_once(self.BackMainPage, 10)

    def StopCapturing(self):
        self.clock_interval.cancel()
        self.capture.stop()

    def BackMainPage(self, dt):
        self.clock_interval.cancel()
        self.capture.stop()
        sm.current = "main_window"

    def Update(self, dt):
        # Prepare camera and get frames (images)
        ret, img = self.capture.read()

        # Dimension of original image
        rows = img.shape[0]
        cols = img.shape[1]

        # Convert BGR color to Gray channel
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Using built-in function detectMultiScale for face detection
        faces = detector.detectMultiScale(gray, 1.05, 5)

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 128, 255), 2)

            self.count_frames = self.count_frames + 1
            # Saving the captured face in the dataset folder RawCapturedPicture
            cv2.imwrite('../RawCapturedPicture/' + str(self.get_name_person) + "_" +
                        str(self.get_id_person) + '_' + str(self.count_frames) + ".jpg", gray[y:y + h, x:x + w])

        img_rotated = cv2.flip(img, 0)
        buf = img_rotated.tobytes()
        image_texture = Texture.create(size=(cols, rows), colorfmt='bgr', bufferfmt='ubyte')
        image_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        # Display image from the texture
        self.texture = image_texture

    def getInfoCaptureForm(self):
        select_screen = self.manager.get_screen("capture_form_window")
        self.get_name_person = select_screen.ids.name_person.text
        self.get_id_person = select_screen.ids.id_person.text
        self.get_ip_cam = select_screen.ids.ip_cam.text


# ============================================================================================ #
class MainWindow(Screen):
    def TrainImages(self):
        """
        Function to train the model, then save the model to the Trainer.yml
        Lastly, it automatically removes all raw images files to reduce the memory capacity.
        :return: None
        """

        if not Tool.showAllFiles('../RawCapturedPicture/'):
            content = Label(text='Could not be trained!')
        else:
            recognizer_train = cv2.face_LBPHFaceRecognizer.create()
            # Get the images from folder, that stores the raw images
            faces, Id = ModelProcessor.imageFileProcessor('../RawCapturedPicture/')

            # Process and save the model to the Trainer.yml
            recognizer_train.train(faces, np.array(Id))
            recognizer_train.save("../ModelTrainer/Trainer.yml")

            content = Label(text='Sucessfully trained!')

            # After training model
            # ..., delete all captured photos
            Tool.deleteContentsDir('../RawCapturedPicture')

        showPopupMessage(content=content)


# ============================================================================================ #
class PopupShowing(Popup):
    pass


# ============================================================================================ #
def showPopupMessage(content):
    # Popup
    pop = Popup(title='Messages',
                content=content,
                size_hint=(None, None),
                size=(400, 400))
    pop.open()


# ============================================================================================ #
class WindowManager(ScreenManager):
    pass


# ============================================================================================ #
# Load Kivy Lang File
kv = Builder.load_file("mymain.kv")
sm = WindowManager()

screens = [MainWindow(name="main_window"), CaptureFormWindow(name="capture_form_window"),
           CaptureWindow(name="capture_window"), TrackFace(name="track_window")]
for screen in screens:
    sm.add_widget(screen)

sm.current = "main_window"


# ============================================================================================ #
class MyMainApp(App):
    def build(self):
        self.title = 'Attendance Management System Application'
        return sm


# ============================================================================================ #
if __name__ == "__main__":
    MyMainApp().run()