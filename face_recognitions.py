import face_recognition
import numpy as np
import cv2
import os
import pickle
from datetime import datetime
import cvzone
import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
from firebase_admin import storage

cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    'databaseURL':"https://attandance-recorder-default-rtdb.firebaseio.com/",
    'storageBucket':"attandance-recorder.appspot.com"
})

bucket = storage.bucket()
cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)

imgBackground = cv2.imread('Resources/background.png')

path = 'image_path'
images = []
classNames = []
myList = os.listdir(path)
# print(myList)
for cls in myList:
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    classNames.append(os.path.splitext(cls)[0])

    fileName = f'{path}/{cls}'
    bucket = storage.bucket()
    blob = bucket.blob(fileName)
    blob.upload_from_filename(fileName)


# Importing the mode images into a list
folderModePath = 'image_path'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

# Load the encoding file
print("Loading Encode File ...")
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
file.close()
encodeListKnown, studentIds = encodeListKnownWithIds
# print(studentIds)
print("Encode File Loaded")
print(classNames)

def findEncoding(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        faceEncode = face_recognition.face_encodings(img)[0]
        encodeList.append(faceEncode)
    return encodeList

def findAttandance(name):
    with open('Attandance.csv','r+') as f:
        myDataList = f.readline()
        myList = []
        for line in myDataList:
            entry = line.split(',')
            myList.append(entry[0])
        if name not in myList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtString}')

encodeList = findEncoding(images)
print("Encoding Complete")

modeType = 0

cap = cv2.VideoCapture(0)
while True:
    sucess, img = cap.read()
    imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    facesFrame = face_recognition.face_locations(imgS)
    encodeFrame = face_recognition.face_encodings(imgS,facesFrame)

    imgBackground[162:162 + 480, 55:55 + 640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[modeType]



    for encodeFace, faceLoc in zip(encodeFrame, facesFrame):
        matches = face_recognition.compare_faces(encodeList, encodeFace)
        faceDis = face_recognition.face_distance(encodeList, encodeFace)
        #print(faceDis)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            #print(name)
            y1,x2,y2,x1 = faceLoc
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1), (x2,y2), (0,255,0),2)
            cv2.rectangle(img, (x1,y2-35),(x2,y2), (0,255,0),cv2.FILLED)
            cv2.putText(img, name,(x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1,(255,255,255),2)
            findAttandance(name)

    cv2.imshow('webcam', img)
    cv2.waitKey(1)