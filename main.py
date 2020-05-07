import numpy as np
import cv2
import pickle

labels={}
with open("labels.pickle",'rb') as fh:
    labels_=pickle.load(fh)
    labels={v:k for k,v in labels_.items()}

recognizer=cv2.face.LBPHFaceRecognizer_create()
face_dataset=cv2.CascadeClassifier('Dataset/haarcascade_frontalface_alt2.xml')
recognizer.read("trained_data.yml")
cap=cv2.VideoCapture(0)

while(1):
    ret,frame=cap.read()
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_dataset.detectMultiScale(gray_frame,scaleFactor=1.5,minNeighbors=5)
    for(x,y,w,h) in faces:
        region=frame[y:y+h,x:x+w]
        gray=gray_frame[y:y+h,x:x+w]
        id,confidence=recognizer.predict(gray)
        if(confidence>=30 and confidence<=100):
            name=labels[id]+" "+str(confidence)[:4]+"%"
            font=cv2.FONT_HERSHEY_COMPLEX
            color=(255,255,255) #White color
            cv2.putText(frame,name,(x-20,y-20),font,1,color,2,cv2.LINE_AA)
        
        cv2.rectangle(frame,(x,y),(x+w,y+h),(250,75,50),4)#frame,starting_coordinates,height and width,color,stroke
    
    cv2.imshow('Image',frame)
    if cv2.waitKey(20) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
