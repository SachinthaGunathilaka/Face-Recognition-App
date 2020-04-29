import glob
from PIL import Image , ImageOps
import numpy as np 
import cv2
import os


def DownSampling(img): 
    vector = []
    # Convert RGB image to Grey-scale
    greyScaleArr = np.array(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) , dtype = float)
    # Down sampling (meka hithala gahapu ekk)
    downSampledArr = greyScaleArr[::5,::5]
    # convert 2D matrix to row matrix
    vector = downSampledArr.reshape(1,downSampledArr.size)
    # maxVal = vector.max()
    # minVal = vector.min()
    
    # mapping between 0-1
    for i in range(len(vector)):
        # vector[i] = (vector[i] - minVal) / (maxVal - minVal)
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

Correctcount = 0
Wrongcount = 0
for testImage in testImages:
    testImageArr = cv2.imread("./Testing/"+testImage)
    minimDistance = 0
    closeClass = ""
    for classFolder in classFolders:
        XiT = getVectors(classFolder+"/*.pgm")
        Xi = XiT.T

        Y = DownSampling(testImageArr)

        # Hi = np.dot(np.dot(Xi , np.linalg.inv(np.dot(XiT,Xi))),XiT)
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
        print(closeClass.replace("./Training/",""),testImage)

print("correct matches : ",Correctcount)
print("wrong matches : ",Wrongcount)




