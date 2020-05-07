import os
import cv2
import pickle
import numpy as np
from PIL import Image

face_dataset=cv2.CascadeClassifier('Dataset/haarcascade_frontalface_default.xml')

y_labels=[]
train_image=[]
id=0
label_id={} #Declare a empty dictionary

for root,dir,files in os.walk("Images/"):
    for filename in files:
        if filename.endswith(".jpg"):
            path=os.path.join(root,filename)
            label=os.path.basename(root).replace(" ","_").lower()
            if not label in label_id:
                label_id[label]=id
                id+=1
            new_id=label_id[label]
            print(label_id)
            img=Image.open(path).convert("L")
            img_array=np.array(img,"uint8")
            #print(img_array)
            faces=face_dataset.detectMultiScale(img_array,scaleFactor=1.5,minNeighbors=5)
            for(x,y,w,h) in faces:
                region=img_array[y:y+h,x:x+w]
                train_image.append(region)
                y_labels.append(new_id)
print(y_labels)
print(train_image)

#For training image and saving trained data
recognizer=cv2.face.LBPHFaceRecognizer_create()
with open("labels.pickle",'wb') as fh:
    pickle.dump(label_id,fh)

recognizer.train(train_image,np.array(y_labels))
recognizer.save("trained_data.yml")