# -*- coding: utf-8 -*-
"""
Created on Sat Apr 22 15:48:40 2023

@author: Shubhi Kant
"""

import cv2
import torch
from torchvision import transforms
import numpy as np
from net import Net
from PIL import Image

# Load the saved model
model_path = "C:\\Users\\Omen\\OneDrive\\Desktop\\Driver-Drowsiness-Detection\\DDN_net.pth"
model = Net()
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the preprocessing steps required by the model
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


face_cascade = cv2.CascadeClassifier(r"C:\Users\Omen\OneDrive\Desktop\Driver-Drowsiness-Detection\haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.4, 5)
    for (x, y, w, h) in faces:
        # Crop the detected face region
        face = Image.fromarray(img[y:y+h, x:x+w, :])
        
        # Preprocess the face image
        face = transform(face)
        
        # Make a prediction using the model
        output = model(face.unsqueeze(0))
        #print(output)
        if output < 0.5:
            prediction = 0
        else:
            prediction = 1
        
        # Display the prediction as text
        if prediction == 1:
            text = "Alert"
        else:
            text = "Drowsy"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        # Draw a rectangle around the detected face
        cv2.rectangle(img, (x,y), (x+w, y+h), (255,0, 0), 2)
    cv2.imshow('img', img)
    k = cv2.waitKey(30) & 0xff
    if k == 2:
        break
    
cap.release()
cv2.destroyAllWindows()