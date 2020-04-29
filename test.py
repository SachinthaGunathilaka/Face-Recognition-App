import os
import numpy as np
import cv2
import sys




image = cv2.imread("./Training/1/2.pgm")


def faceDetector(image):
    width = 80
    height = 80
    dim = (width, height)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    print(gray)

    faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    roi_color = np.array([[]],dtype = float)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_color = image[y:y + h, x:x + w]
        # cv2.imwrite(str(w) + str(h) + '_faces.jpg', roi_color)
    roi_color = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)
    print(roi_color)
    # print(roi_color)
    resized = np.array(cv2.resize(roi_color, dim, interpolation = cv2.INTER_AREA),dtype = float)
    cv2.imwrite('m.jpg', resized)

    # print(resized)


faceDetector(image)

