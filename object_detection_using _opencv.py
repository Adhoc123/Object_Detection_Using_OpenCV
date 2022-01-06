#!/usr/bin/env python
# coding: utf-8

# In[3]:


import cv2                        #importing opencv for image processing


# In[4]:


import matplotlib.pyplot as plt   #importing matplotlib for data visualization


# In[5]:


config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'   #loading configuration file
frozen_model = 'frozen_inference_graph.pb'                     #loading tensorflow pretrained model


# In[6]:


model = cv2.dnn_DetectionModel(frozen_model,config_file)       #loading tensorflow pretrained model into memory


# In[7]:


classLabels = []                                           #creating a list
file_name = 'Labels.txt'                                   #Reading labels
with open(file_name,'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')      #pushing into list


# In[8]:


print(classLabels)                                                #printing classlabels


# In[9]:


print(len(classLabels))                                            #printing total class length


# In[10]:


model.setInputSize(320,320)                                      #setting input size
model.setInputScale(1.0/127.5)    #255/2=127.5                   #scaling input
model.setInputMean((127.5,127.5,127.5))                   #setting input mean
model.setInputSwapRB(True)                  #setting input swap = True for automatic RGB conversion


# In[ ]:


cap = cv2.VideoCapture(1)          #for capturing webcam

if not cap.isOpened():             #checking if the video is opened correctly
    cap = cv2.VideoCapture(0) 
if not cap.isOpened():
    raise IOError('Cannot open webcam')

font_scale = 3                    #font size
font = cv2.FONT_HERSHEY_PLAIN     #font type

while True:
    ret,frame = cap.read()
    ClassIndex, confidece, bbox = model.detect(frame,confThreshold=.55)      #for 50% confidece
    print(ClassIndex)
    if(len(ClassIndex)!=0):
        for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):      #flattening and zipping | var>=3
            if(ClassInd<=80):
                cv2.rectangle(frame,boxes,(255,0,0),2)   #plot this box having bgr
                cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40),font,fontScale=font_scale,color=(0,255,0),thickness=3)
    cv2.imshow('Object Detection',frame)
    if cv2.waitKey(2) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()


# In[ ]:




