import glob
from PIL import Image , ImageOps , ImageEnhance
import numpy as np 
from cv2 import *
import cv2
import os



def apply_brightness_contrast(input_img, brightness=0, contrast=0):
    
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow)/255
        gamma_b = shadow

        buf = cv2.addWeighted(input_img, alpha_b, input_img, 0, gamma_b)
    else:
        buf = input_img.copy()

    if contrast != 0:
        f = 131*(contrast + 127)/(127*(131-contrast))
        alpha_c = f
        gamma_c = 127*(1-f)

        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)

    return buf







faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
def DownSampling(img): 
    vector = []
    greyScaleImg = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) , dtype = float)

    enhancedImg = np.array(apply_brightness_contrast(greyScaleImg , 100 , -50) , dtype = float )

    # downSampledImg1 = cv2.blur(enhancedImg,(3,3))


    # downSampledImg1 = cv2.threshold(enhancedImg,127,255,cv2.THRESH_BINARY)
    downSampledImg = enhancedImg[::5,::5]
    downSampledImg = downSampledImg[::3,::3]
    
    vector = downSampledImg.reshape(1,downSampledImg.size)

    for i in range(len(vector)):
        vector[i] = vector[i] / 255.0
    return vector


# when training file is given return vectors(Xi)
def getVectors(file_path):
    images = [cv2.imread(file) for file in glob.glob(file_path)]

    vectors = np.array([])
    for img in images:
        vector = DownSampling(img)
        
        if(vectors.size == 0):
            vectors = vector
        else:
            vectors = np.append(vectors,vector,axis=0)

    return vectors


classFolders = []
for root, dirs, files in os.walk("./Training"):
    if(root != "./Training"):
        classFolders.append(root)



testImages = []
for root, dirs, files in os.walk("./Testing"):
    for name in files:
        if name.endswith((".pgm")):
            testImages.append(name)


XiTs = []
for classFolder in classFolders:
    XiT = getVectors(classFolder+"/*.pgm")
    XiTs.append([XiT,classFolder])




Correctcount = 0
Wrongcount = 0
for testImage in testImages:
    testImageArr = cv2.imread("./Testing/"+testImage)
    minimDistance = 0
    closeClass = ""
    for XiT , classFolder in XiTs:
    
        Xi = XiT.T
        Y = DownSampling(testImageArr)

        h0 = np.dot(XiT,Xi)

        h1 = np.linalg.inv(h0)
        h2 = np.dot(Xi , h1)
        Hi = np.dot(h2,XiT)

        Yi = Hi*Y

        distance = np.linalg.norm(Yi - Y)
        
        if(closeClass == ""):
            minimDistance = distance
            closeClass = classFolder
            
        elif(distance <= minimDistance):
            minimDistance = distance
            closeClass = classFolder
            

    # print(closeClass.replace("./Training/",""),testImage)
    a = closeClass.replace("./Training/","")
    b = testImage
    alen = len(a)
    if(a == b[:alen]):

        Correctcount = Correctcount +1
    else:
        Wrongcount = Wrongcount +1
        # print(closeClass.replace("./Training/",""),testImage)

        print(closeClass.replace("./Training/",""),testImage)

print("correct matches : ",Correctcount)
print("wrong matches : ",Wrongcount)




