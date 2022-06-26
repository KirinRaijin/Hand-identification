# -*- coding: utf-8 -*-
"""
Created on Sat Jun 25 15:17:11 2022

@author: rishi
"""

import cv2
import time
import os
import mediapipe as mp
wcam,hcam=1240,720

cap=cv2.VideoCapture(0)
cap.set(3,wcam)
cap.set(4,hcam)

folderPath="FingerImages"
myList =os.listdir(folderPath)
print(myList)
overlayL= []
for imPath in myList:
    image=cv2.imread(f'{folderPath}/{imPath}')
    overlayL.append(image)
print(len(overlayL))
mpHands=mp.solutions.hands
hands=mpHands.Hands()
mpDraw=mp.solutions.drawing_utils
pTime=0

tipIds=[4,8,12,16,20]
while True:
    success,img=cap.read()
    imgRGB=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    lmList = []
    handsType=[]
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                #print(id,lm)
                h, w, c=img.shape
                cx, cy=int(lm.x*w),int(lm.y*h)
                lmList.append([id, cx, cy])
            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
        handsType=[]
        if results.multi_hand_landmarks!=None:
            for hand in results.multi_handedness:
                handType=hand.classification[0].label
                handsType.append(handType)
                #print(id, cx,cy)
                """if id==0:
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)"""
        print(handsType)    
    fingersL=[]
    fingersR=[]
    if results.multi_hand_landmarks != None:
        for i in handsType:
               if i=='Right' and lmList[tipIds[0]][1]<lmList[tipIds[0]-1][1]:
                  fingersL.append(1)
               elif i=='Right':
                  fingersL.append(0)
                    
               if len(lmList) !=0:
                 for id in range(1,5):
                    if i=='Right' and lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                        fingersL.append(1)
                    elif i=='Right':
                        fingersL.append(0)
               
               if i=='Left' and lmList[tipIds[0]][1]>lmList[tipIds[0]-1][1]:
                  fingersR.append(1)
               elif i=='Left':
                  fingersR.append(0)
                    
               if len(lmList) !=0:
                 for id in range(1,5):
                    if i=='Left' and lmList[tipIds[id]][2]<lmList[tipIds[id]-2][2]:
                        fingersR.append(1)
                    elif i=='Left':
                        fingersR.append(0)
               print(fingersL)
               print(fingersR)
    sum=0
    for i in fingersL:
        sum+=i
    for i in fingersR:
        sum+=i
    
        
    
    
    h,w,c=overlayL[sum].shape
    img[0:h,0:w]=overlayL[sum]
    
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv2.putText(img,f'FPS:{int(fps)}',(300,90),cv2.FONT_HERSHEY_COMPLEX,3,(255,0,0),3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)