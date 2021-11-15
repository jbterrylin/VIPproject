#!/usr/bin/env python
# coding: utf-8

# In[20]:


import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
# from tensorflow import keras
import cv2
import os
from streamlit_cropper import st_cropper
from PIL import Image
import shutil
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer, RTCConfiguration
import av


# shutil.unpack_archive("./Freeze_BestModelAge.zip", "./")
# shutil.unpack_archive("./GenderPrediction.zip", "./")
# from matplotlib import pyplot as plt

# https://stackoverflow.com/questions/14134892/convert-image-from-pil-to-opencv-format
# pil_image = PIL.Image.open('Image.jpg').convert('RGB') 
# open_cv_image = numpy.array(pil_image) 
# # Convert RGB to BGR 
# open_cv_image = open_cv_image[:, :, ::-1].copy()

# RTC_CONFIGURATION = RTCConfiguration(
#     {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
# )


# In[9]:


# https://towardsdatascience.com/face-detection-in-2-minutes-using-opencv-python-90f89d7c0f81
def get_heads(img):
    heads = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    # Draw rectangle around the faces
    for (x, y, w, h) in faces:
        size = 0
        if (w > h):
            size = w
        else:
            size = h
        newx = int(x-(w*0.1))
        neww = int(w*1.2)
        newy = int(y-(h*0.3))
        newh = int(h*1.4)
#         cv2.rectangle(img, (newx, newy), (newx+neww, newy+newh), (255, 0, 0), 2)
#         cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        heads.append(img[
            int(newy):int(newy+newh),
            int(newx):int(newx+neww) 
        ])
    return heads


# In[ ]:


# https://docs.streamlit.io/library/advanced-features/session-state
if 'img' not in st.session_state:
    st.session_state.img = []
    
if 'heads' not in st.session_state:
    st.session_state.heads = []
    
if 'headsbool' not in st.session_state:
    st.session_state.headsbool = []


# In[ ]:


progress = st.sidebar.radio("Progress",('Upload Image', 'Process Image', 'Result'))


# In[ ]:


class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.i = 0
        self.faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml')
        self.age_model = tf.keras.models.load_model('./model/AgeDetection/Freeze/Freeze_BestModelAge.h5')
        self.labels_age = {0: 'Adolescence', 1: 'Adult',2:'Child',3:'Senior Citizen'}
        self.gender_model = tf.keras.models.load_model('./model/GenderDetection/Freeze/GenderPrediction.h5')
        self.labels_gender = {0: 'Female', 1: 'Male'}

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 5)
        i =self.i+1
        for (x, y, w, h) in faces:
            head = img[
                int(y):int(y+h),
                int(x):int(x+w) 
            ]
            head = head/255
            resized = cv2.resize(head, (80,80))
            reshaped = resized.reshape(1,80, 80,3)
            predictions = self.age_model.predict(reshaped)
            predicted_class = np.argmax(predictions,axis=1).item(0)
            age_predicted_label = self.labels_age[predicted_class]
            
            predictions = self.gender_model.predict(reshaped)
            predicted_class = np.argmax(predictions,axis=1).item(0)
            gender_predicted_label = self.labels_gender[predicted_class]
            
            cv2.rectangle(img, (x, y), (x + w, y + h), (95, 207, 30), 3)
            cv2.rectangle(img, (x, y - 40), (x + w, y), (95, 207, 30), -1)
            cv2.putText(img, age_predicted_label + "(" + gender_predicted_label + ")", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        return img

if(progress == 'Upload Image'):
    st.title("Upload Image")
    uploadmethod = st.radio("Way to upload",('Upload Image', 'With Camera'))
    if(uploadmethod == 'Upload Image'):
        file = st.file_uploader(label='Upload file', type=['png', 'jpg'])
        if (file):
            image = Image.open(file)
            img_array = np.array(image)
            st.session_state.img = img_array
        
    if(uploadmethod == 'With Camera'):
        webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
#         run = True
#         placeholder = st.empty()
#         FRAME_WINDOW = st.image([])
#         camera = cv2.VideoCapture(0)

#         frame = camera.read()
#         if placeholder.button('Take Picture'):
#             st.session_state.img = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
#             placeholder.empty()
#             run = False
#         while run:
#             _, frame = camera.read()
#             frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#             FRAME_WINDOW.image(frame)
#         if (len(img) != 0):
#             retakebtn = st.button("Retake")
#             if retakebtn:
#                 run =True
#                 st.session_state.img = []

    if (len(st.session_state.img) != 0):
        st.write("Click “Process Image” radio button on the side bar when the image is uploaded")
        st.image(st.session_state.img)
        
        st.session_state.heads = get_heads(st.session_state.img)
        for i in range(len(get_heads(st.session_state.img))):
            st.session_state.headsbool.append(True)


# In[ ]:


if(progress == 'Process Image'):
    st.title("Process Image")
    if (len(st.session_state.img) == 0):
        st.error('No Image choosed')
    else:
        st.write("Number of face detected:", len(st.session_state.heads))
        st.write("detected face have been load to choosed image")
        
        box_color = st.color_picker(label="Box Color", value='#0000FF')
        img = Image.fromarray(st.session_state.img)
        # Get a cropped image from the frontend
        cropped_img = st_cropper(img, realtime_update=True, box_color=box_color,
                                    aspect_ratio=(1,1))
        savebtn = st.button("save")
        if savebtn:
            st.session_state.heads.append(cv2.cvtColor(np.array(cropped_img)[:, :, ::-1].copy(), cv2.COLOR_RGB2BGR))
            st.session_state.headsbool.append(True)
        # Manipulate cropped image at will
        st.header("Preview")
        _ = cropped_img.thumbnail((150,150))
        st.image(cropped_img)
        
        st.header("Choosed head")
        for i in range(len(st.session_state.heads)):
            if (st.session_state.headsbool[i] == True):
                placeholder = st.empty()
                col1, col2, col3 = placeholder.columns(3)
                with col1:
                    st.image(st.session_state.heads[i])
                with col2:
                    if st.button("Delete",key = "Delete_"+str(i)):
                        placeholder.empty()
                        st.session_state.headsbool[i] = False


# In[29]:


if(progress == 'Result'):
    st.title("Result")
    if (len(st.session_state.img) == 0):
        st.error('No Image choosed')
    if (np.count_nonzero(st.session_state.headsbool) == 0):
        st.error('Picture not processed and cannot detect face by CascadeClassifier')
    else:
        st.image(st.session_state.img)
        age_model = tf.keras.models.load_model('./model/AgeDetection/Freeze/Freeze_BestModelAge.h5')
        labels_age = {0: 'Adolescence', 1: 'Adult',2:'Child',3:'Senior Citizen'}
        gender_model = tf.keras.models.load_model('./model/GenderDetection/Freeze/GenderPrediction.h5')
        labels_gender = {0: 'Female', 1: 'Male'}
        for i in range(len(st.session_state.heads)):
            if (st.session_state.headsbool[i] == True):
                col1, col2 = st.columns([1, 2])
                img = st.session_state.heads[i]
                img = img/255
                resized = cv2.resize(img, (80,80))
                reshaped = resized.reshape(1,80, 80,3)
                
                predictions = age_model.predict(reshaped)
                predicted_class = np.argmax(predictions,axis=1).item(0)
                age_predicted_label = labels_age[predicted_class]
                age_df = pd.DataFrame(predictions, columns=[labels_age[key] for key in labels_age])
                
                predictions = gender_model.predict(reshaped)
                predicted_class = np.argmax(predictions,axis=1).item(0)
                gender_predicted_label = labels_gender[predicted_class]
                gender_df = pd.DataFrame(predictions, columns=[labels_gender[key] for key in labels_gender])
                with col1:
                    st.image(st.session_state.heads[i])
                with col2:
#                     st.write(age_predicted_label)
                    with st.expander(age_predicted_label):
                        st.write(age_df)
                    with st.expander(gender_predicted_label):
                        st.write(gender_df)
#                     st.write(gender_predicted_label)

