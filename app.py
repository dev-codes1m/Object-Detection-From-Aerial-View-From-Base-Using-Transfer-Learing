import flask
from keras.models import load_model
import cv2
from keras.preprocessing import image
from  keras.preprocessing.image import img_to_array
from PIL import Image as im
from flask import Flask,request,url_for,render_template,jsonify,app
import numpy as np
import pandas as pd
import os
os. environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import send_file
app = Flask(__name__)

model = load_model('ODFAV.h5')
@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_object',methods = ['POST'])
def predictobject():
    img = str(request.values.get('image'))
    images = cv2.imread(img)
    original = images.copy()
    gray = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    dilate = cv2.dilate(thresh, kernel, iterations=1)

    # Find contours, obtain bounding box coordinates, and extract ROI
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    image_number = 0
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)

        ROI = original[y:y + h, x:x + w]
        #     ROI = np.arange(0, 737280, 1, np.uint8)
        i1 = im.fromarray(ROI)
        i1 = i1.resize((224, 224))
        i1 = image.img_to_array(i1)
        i1 = np.expand_dims(i1, axis=0)
        i1 = i1 / 255.0
        k = model.predict(i1)
        if k[0][0] == np.max(k):
            cv2.putText(images, 'CAR', (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.75, (0, 255, 0), 1)
            cv2.rectangle(images, (x, y), (x + w, y + h), (0, 255, 0), 2)
        elif k[0][2] == np.max(k):
            cv2.putText(images, 'People', (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.75, (255, 255, 0), 1)
            cv2.rectangle(images, (x, y), (x + w, y + h), (255, 255, 0), 2)
        elif k[0][3] == np.max(k):
            cv2.putText(images, 'Trash', (x + w, y + h), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
            cv2.rectangle(images, (x, y), (x + w, y + h), (0, 0, 255), 2)

        #     cv2.imwrite("ROI_{}.png".format(image_number), ROI)
        #     print(ROI)

        image_number += 1

    # cv2.imshow('image', images)
    cv2.imwrite('Uploads/Final_result.jpg', images)
    # # cv2.imshow('thresh', thresh)
    # # cv2.imshow('dilate', dilate)
    # cv2.waitKey()


    return render_template("home.html")
    

if __name__== "__main__":
    app.run(debug=True,host="0.0.0.0",port=8080)