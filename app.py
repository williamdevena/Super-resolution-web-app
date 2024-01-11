import base64

import cv2
from flask import Flask, render_template, request
from torchsr.models import ninasr_b0, ninasr_b1, ninasr_b2

from src import image_processing
from src.enhancer import Enhancer

app = Flask(__name__)

@app.route('/')
@app.route('/home')
def home() -> str:
    return render_template('index.html', uploaded_image=None)


@app.route('/upload', methods=['POST'])
def upload() -> str:
    #MODEL = ninasr_b0
    #MODEL = ninasr_b2
    model = ninasr_b1
    scale = int(request.form['scale'])

    filestr = request.files['image'].read()
    image = image_processing.imageFromString(filestr=filestr)
    image = cv2.cvtColor(image , cv2.COLOR_BGR2RGB)
    enhancer = Enhancer(model=model, scale=scale)
    enhanced_image = enhancer.enhance(image=image) * 255

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