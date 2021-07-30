import cv2
import numpy as np
import os
import threading
from playsound import playsound
import time
n_face_detected =0
siren =0
def thread_function(_):
    global siren
    while siren==1:
        playsound('1.mp3')

def face_detect(frame,model):
    global n_face_detected
    face_cascade =cv2.CascadeClassifier(model)
    frame_gray =cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces =face_cascade.detectMultiScale(frame_gray,1.1,4)
    for (x,y,w,h) in faces:
        n_face_detected =n_face_detected+1
        frame_croped =frame[y:y+w,x:x+h,:]
        frame_croped =cv2.resize(frame_croped,(300,300))
        path ='C:/Users/User-asus/Documents/ThiefCatcher'
        cv2.imwrite(os.path.join(path,f'theifface{n_face_detected}.jpg'),frame_croped)
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),2)
        cv2.putText(frame,'face detected',(x,y+w+10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    return frame

def Frame_difference(frame1,frame2):
    frame2 =frame2.copy()
    global n_face_detected
    diff =cv2.absdiff(frame1,frame2)
    gray =cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
    blured =cv2.GaussianBlur(gray,(5,5),0)
    edge_detect =cv2.Canny(blured,240,255)
    dilated =cv2.dilate(edge_detect,None,iterations=15)    
    return dilated

def moving(frame,f_difference):
    frame =frame.copy()
    countours , hierarchy =cv2.findContours(f_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in countours:
        x,y,w,h =cv2.boundingRect(c)
        if cv2.contourArea(c)>5000 :
            cv2.putText(frame,'status : movement',(30,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,0,255),2,cv2.LINE_AA)
    return frame,len(countours)>0

def missing(frame,f_difference):
    frame =frame.copy()
    countours , hierarchy =cv2.findContours(f_difference,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    for c in countours:
        x,y,w,h =cv2.boundingRect(c)
        cv2.rectangle(frame,(x,y),(x+h,y+w),(0,0,255),3)
        cv2.putText(frame,'somthing is missing',(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),1,cv2.LINE_AA)
        print('somthing is missing')
    return frame

cap =cv2.VideoCapture(0)
_,first_frame =cap.read()
_,frame2 =cap.read()
_,frame3 =cap.read()
x = threading.Thread(target=thread_function, args=(1,))
while(cap.isOpened()):
    frame_diff_moving =Frame_difference(frame3,frame2)
    frame_diff_missing =Frame_difference(first_frame,frame2)
    processed_frame,status =moving(frame2,frame_diff_moving)
    if status==0 :processed_frame =missing(frame2,frame_diff_missing)
    if status==1 and siren==0:
        siren =1
        x.start()
    processed_frame =face_detect(processed_frame,'haarcascade_frontalface_default.xml')
    #model path
    cv2.imshow('ThiefCatcher',processed_frame)
    cv2.imshow('diff',frame_diff_moving)
    frame3 =frame2.copy()
    _,frame2 =cap.read()
    frame2 =cv2.flip(frame2, 1)
    if(cv2.waitKey(1) & 0xFF == ord('q')):
        siren=0
        break    
siren=0
cv2.destroyAllWindows()