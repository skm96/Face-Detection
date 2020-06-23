# -*- coding: utf-8 -*-
"""
Created on Tue Jun 23 20:16:57 2020

@author: Soumya Kanti Mandal
"""

import cv2
import urllib.request
from matplotlib import pyplot as plt
from pylab import rcParams
image_url="https://skm96.github.io/image/s2.jpg" 
image_name="skm.jpg"
urllib.request.urlretrieve(image_url, image_name)
image1 = cv2.imread("skm.jpg")
plt.imshow(image1)

# now fixing color axis and increse size
def plt_show(image,title="",gray=False,size=(12,10)):
    temp = image
    #fix color
    if gray==False:
        temp=cv2.cvtColor(temp,cv2.COLOR_BGR2RGB)
    #change img size
    rcParams['figure.figsize'] = size[0] , size[1]
    plt.axis("off")
    plt.title(title)
    plt.imshow(temp,cmap='gray')
    plt.show()

#display image nicely    
plt_show(image1,"Face Detection Phase 1")


# image detect using Haar Cascades
haaarcascade_url="https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
haar_name="haarcascade_frontalface_defult.xml"
urllib.request.urlretrieve(haaarcascade_url,haar_name)

#create facial detection classifier  
detector=cv2.CascadeClassifier('haarcascade_frontalface_defult.xml')


# detect faces using the array 
faces_list = detector.detectMultiScale(image1,
                                       scaleFactor=1.2,
                                       minNeighbors=10,
                                       minSize=(64,64),
                                       flags=cv2.CASCADE_SCALE_IMAGE)
print(faces_list)

# drow ractangles around faces to check
for face in faces_list:  #draw ractangle for each face
    #x & y axis and height & width
    (x,y,w,h) = face
    cv2.rectangle(image1,
                  (x,y), #bottom left corner as like axis
                  (x+w,y+h), #top right corner 
                  (0,255,0), #color green 
                  3)  #line thickness of ractangle
plt_show(image1)

    
# we now see that we can only plot faces of diffrent faces 
for face in faces_list:
    (x,y,w,h) = face
    #plot each face now
    face=image1[y:y + h, x:x + w] #crop to face
    face_resize =cv2.resize(face,(80,80)) #resize
    plt_show(face_resize,size=(6,5))
    
    
    
    

