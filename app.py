import base64
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, render_template, request, url_for
from flask.helpers import send_file
from torchsr.models import ninasr_b0

from src.enhancer import Enhancer

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html', uploaded_image=None)

@app.route('/upload', methods=['POST'])
def upload():
    SCALE = 3
    RESCALING_FACTOR = 0.1
    MODEL = ninasr_b0
    filestr = request.files['image'].read()
    file_bytes = np.fromstring(filestr, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    width = int(image.shape[1] * RESCALING_FACTOR)
    height = int(image.shape[0] * RESCALING_FACTOR)
    dim = (width, height)
    image = cv2.resize(image, dim)
    #print(image)
    enhancer = Enhancer(model=MODEL, scale=SCALE)
    enhanced_image = enhancer.enhance(image=image) * 255

    #print(image)

    enhanced_image = cv2.cvtColor(enhanced_image, cv2.COLOR_RGB2BGR)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    _, enhanced_encoded = cv2.imencode('.jpg', enhanced_image)
    _, original_encoded = cv2.imencode('.jpg', image)

    enhanced_string = enhanced_encoded.tostring()
    original_string = original_encoded.tostring()

    uploaded_image_base64 = base64.b64encode(original_string).decode('utf-8')

    enhanced_image_base64 = base64.b64encode(enhanced_string).decode('utf-8')

    return render_template('index.html', uploaded_image_base64=uploaded_image_base64, enhanced_image_base64=enhanced_image_base64)




if __name__ == '__main__':
    app.run(debug=True)