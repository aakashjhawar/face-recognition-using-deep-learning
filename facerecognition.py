#!/usr/bin/env python3

"""
BASED ON https://github.com/aakashjhawar/face-recognition-using-deep-learning

Change log:


- Added a method to create a directory for the user.
- Added a method to implement the dataset for face recognition.
- Added a progress bar to the method to implement the dataset.
- Added a method to train the model.
- Added a method to recognize the face.
- Added a method to extract the face.

"""

import time

import cv2
import imutils
from imutils import paths
import numpy as np
import os
import pickle

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


percent = 0


class FaceRecognition:
    def __init__(self, cameraIndex):
        """
        Initialize the class.
        Required argument: cameraIndex => 0 if you have only one camera, test if more.
        Warning: HDMI displays are considered as cameras (wth?).
        Returns nothing.
        """
        self.cameraIndex = int(cameraIndex)

        # Using pycharm, you have to run from terminal or paths doesn't work
        self.dataset = 'dataset/'
        self.embeddings = './output/embeddings.pickle'
        self.recognizer = './output/recognizer.pickle'
        self.le = './output/le.pickle'
        self.protopath = "./face_detection_model/deploy.prototxt"
        self.deploypath = "./face_detection_model/deploy.prototxt"
        self.modelpath = "./face_detection_model/res10_300x300_ssd_iter_140000.caffemodel"
        self.detector = cv2.dnn.readNetFromCaffe(self.protopath, self.modelpath)
        self.embedder = cv2.dnn.readNetFromTorch("./openface_nn4.small2.v1.t7")

        self.knownEmbeddings = []
        self.knownNames = []
        self.total = 0

    def extract(self):
        """
        Extract the faces from the dataset's pictures.
        Save the embeddings and the names in a pickle file.
        Returns nothing, no arguments required.
        """
        print("Loading Face Detection Model...")
        print("Quantifying Faces...")
        image_paths = list(paths.list_images(self.dataset))

        for (i, imagePath) in enumerate(image_paths):
            if i % 50 == 0:
                print("Processing image {}/{}".format(i, len(image_paths)))

            name = imagePath.split(os.path.sep)[-2]
            image = cv2.imread(imagePath)
            image = imutils.resize(image, width=600)
            (h, w) = image.shape[:2]

            image_blob = cv2.dnn.blobFromImage(
                cv2.resize(image, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False)

            self.detector.setInput(image_blob)
            detections = self.detector.forward()

            if len(detections) > 0:
                i = np.argmax(detections[0, 0, :, 2])
                confidence = detections[0, 0, i, 2]

                if confidence > 0.5:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    face = image[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    if fW < 20 or fH < 20:
                        continue

                    face_blob = cv2.dnn.blobFromImage(face, 1.0 / 255,
                                                      (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    self.embedder.setInput(face_blob)
                    vec = self.embedder.forward()

                    self.knownNames.append(name)
                    self.knownEmbeddings.append(vec.flatten())
                    self.total += 1

        print("[INFO] Serializing {} encodings...".format(self.total))
        data = {"embeddings": self.knownEmbeddings, "names": self.knownNames}
        with open(self.embeddings, "wb") as f:
            f.write(pickle.dumps(data))

        print("[INFO] Done")

    def create_directory(self, username):
        """
        Create a directory for the user.
        Required argument: username
        Returns nothing.
        """
        directory = username
        path = os.path.join(self.dataset, directory)

        if not os.path.exists(path):
            try:
                os.makedirs(path)
                print(f"Directory '{path}' created successfully.")
            except OSError as e:
                print(f"Error creating directory '{path}': {e}")
        else:
            print(f"Directory '{path}' already exists. And ready to use !")

    def implement_dataset(self, username):
        """
        Implement the dataset for face recognition : takes 60 pictures of the user in 30sec.
        Required argument: username
        Returns nothing.
        """

        directory = username
        path = os.path.join(self.dataset, directory)

        self.create_directory(username)

        cap = cv2.VideoCapture(self.cameraIndex)

        for i in range(1, 61):
            ret, frame = cap.read()
            cv2.imwrite(f"{path}/{i}.png", frame)

            percent_done = round((i / 60) * 100)
            done = round(percent_done / (100 / 60))
            togo = 60 - done
            done_str = '█' * int(done)
            togo_str = '░' * int(togo)

            print(f'\t⏳ Implementing dataset: [{done_str}{togo_str}] {percent_done}% done', end='\r')

            time.sleep(0.2)

            cv2.waitKey(1)

        print(f'\t✅ Implementing dataset: [DONE]')
        cap.release()
        cv2.destroyAllWindows()

    def train(self):
        """
        Train the model.
        Returns nothing, no arguments required.
        """

        # dump the facial embeddings + names to disk
        print("[INFO] serializing {} encodings...".format(self.total))
        data = {"embeddings": self.knownEmbeddings, "names": self.knownNames}
        f = open(self.embeddings, "wb")
        f.write(pickle.dumps(data))
        f.close()

        print("[INFO] Done")

        # Load face embeddings
        print("[INFO] loading embeddings...")
        data = pickle.loads(open(self.embeddings, "rb").read())

        # encode the labels
        print("[INFO] encoding labels...")
        le = LabelEncoder()
        labels = le.fit_transform(data["names"])

        # train the model used to accept the 128-d embeddings of the face and
        # then produce the actual face recognition
        print("[INFO] training model...")
        recognizer = SVC(C=1.0, kernel="linear", probability=True)
        recognizer.fit(data["embeddings"], labels)

        # write the actual face recognition model to disk
        f = open(self.recognizer, "wb")
        f.write(pickle.dumps(recognizer))
        f.close()

        # write the label encoder to disk
        f = open(self.le, "wb")
        f.write(pickle.dumps(le))
        f.close()

        print('[INFO] model trained !')

    def recognition(self):
        """
        Recognize the face.
        Returns the name and the probability of the recognition in a tuple.
        No arguments required.
        """

        # Load the serialized face detector
        global percent
        print("Loading face detector...")

        detector = cv2.dnn.readNetFromCaffe(self.deploypath, self.modelpath)

        # Load the serialized face feature extractor model
        print("Loading face recognizer...")
        extractor = self.embedder

        # Load the actual face recognition model along with the label encoder
        recognizer = pickle.loads(open(self.recognizer, "rb").read())
        le = pickle.loads(open(self.le, "rb").read())

        # Initialize video capture from the default camera (index 0)
        cap = cv2.VideoCapture(self.cameraIndex)

        name = None
        proba = 0

        print("Recognition started")
        # Loop over the frames from the video stream
        while True:
            # Capture the frame from the camera
            _, frame = cap.read()

            if not _:
                print("Failed to grab frame")
                break
            # Resize the frame to have a width of 600 pixels (maintaining the aspect ratio),
            # then get the dimensions of the frame
            frame = imutils.resize(frame, width=600)
            (h, w) = frame.shape[:2]

            # Construct a blob from the frame
            blob_image = cv2.dnn.blobFromImage(
                cv2.resize(frame, (300, 300)), 1.0, (300, 300),
                (104.0, 177.0, 123.0), swapRB=False, crop=False
            )

            # Apply the deep learning-based face detector to localize faces in the input image
            detector.setInput(blob_image)
            detections = detector.forward()

            # Loop over the detections
            for i in range(0, detections.shape[2]):
                # Extract the confidence associated with the prediction
                confidence = detections[0, 0, i, 2]

                # Filter out weak detections
                if confidence > 0.5:
                    # Calculate the (x, y)-coordinates of the bounding box for the face
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # Extract the face ROI
                    face = frame[startY:endY, startX:endX]
                    (fH, fW) = face.shape[:2]

                    # Ensure the face width and height are sufficiently large
                    if fW < 20 or fH < 20:
                        continue

                    # Construct a blob for the face ROI, then pass the blob through our face feature extractor model to
                    # obtain the 128-d quantification of the face
                    blob_face = cv2.dnn.blobFromImage(face, 1.0 / 255, (96, 96), (0, 0, 0), swapRB=True, crop=False)
                    extractor.setInput(blob_face)
                    vec = extractor.forward()

                    # Perform classification to recognize the face
                    predictions = recognizer.predict_proba(vec)[0]
                    j = np.argmax(predictions)
                    proba = predictions[j]
                    name = le.classes_[j]

                    percent = round(proba * 100)

            if proba > 0.8:
                if name is not None and name != "unknown":
                    return name, percent

                elif name == "unknown":
                    pass

        # clean up
        cap.release()


# Testing the class
if __name__ == "__main__":
    face_recognition = FaceRecognition(cameraIndex=0)
    # face_recognition.implement_dataset(username='test')
    face_recognition.extract()
    face_recognition.train()
    print(face_recognition.recognition())