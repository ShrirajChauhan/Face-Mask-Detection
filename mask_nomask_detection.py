import cv2
import tensorflow as tf
import numpy as np



face_cascade=cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
model=tf.keras.models.load_model('saved_model.h5')

vid=cv2.VideoCapture(0)

while(True):
    _,frame=vid.read()
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,1.5,8)

    
    for (x,y,w,h) in faces:
        face_frame=rgb[y:y+w,x:x+h]
        face_frame=cv2.resize(face_frame,(256,256))
        face_frame=np.reshape(face_frame,(1,256,256,3))
        prediction=model.predict(face_frame)
        color=(0,255,0) if prediction[0][0] > prediction[0][1] else (0,0,255)
        label='Mask' if prediction[0][0] > prediction[0][1] else 'No Mask'
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,3)
        cv2.putText(frame,label,(x,y-20),cv2.FONT_HERSHEY_COMPLEX,2,color,2)
    cv2.imshow('VID',frame)
        
        
    k=cv2.waitKey(1) & 0xFF
    if k==27:
        break
vid.release()
cv2.destroyAllWindows()