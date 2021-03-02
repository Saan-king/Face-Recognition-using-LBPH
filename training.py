import numpy as np
import cv2
import os

import face_recognition as fr
print(fr)

test_img=cv2.imread(r'C:\Users\msath\Desktop\LBPH\pykara.jpg')

faces_detected,gray_img=fr.faceDetection(test_img)
print("Face Detected: ",faces_detected)

#Training 

faces,faceID=fr.labels_for_training_data(r'C:\Users\msath\Desktop\LBPH\images')
face_recognizer=fr.train_classifier(faces,faceID)
face_recognizer.save(r'C:\Users\msath\Desktop\LBPH\trainingData.yml')

name={0:'Saan'}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+h,x:x+h]
    label,confidence=face_recognizer.predict(roi_gray)
    print("Confidence: ",confidence)
    print("Label: ",label)
    fr.draw_rect(test_img,face)
    predicted_name=name[label]
    fr.put_text(test_img,predicted_name,x,y)

resized_img=cv2.resize(test_img,(1000,700))

cv2.imshow("face detection: ",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()