from flask import Flask, request, render_template, send_file
import cv2
from PIL import Image
import numpy as np
import os
from io import BytesIO

app = Flask(__name__)
UPLOAD_FOLDER = 'static'
RESULT_PATH = os.path.join(UPLOAD_FOLDER, 'result.jpg')

def create_bw_sketch(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    inverted = 255 - gray
    blurred = cv2.GaussianBlur(inverted, (21, 21), 0)
    inverted_blur = 255 - blurred
    sketch = cv2.divide(gray, inverted_blur, scale=256.0)
    return Image.fromarray(sketch)

def create_color_sketch(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray, color = cv2.pencilSketch(img, sigma_s=60, sigma_r=0.07, shade_factor=0.05)
    return Image.fromarray(color)

def create_cartoon_effect(pil_image):
    img = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray_blur, 255,
                                  cv2.ADAPTIVE_THRESH_MEAN_C,
                                  cv2.THRESH_BINARY, 9, 9)
    color = cv2.bilateralFilter(img, 9, 300, 300)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return Image.fromarray(cartoon)

def create_oil_effect(pil_image):
    img = np.array(pil_image.convert("RGB"))
    oil = cv2.xphoto.oilPainting(img, 7, 1)
    return Image.fromarray(oil)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/sketch', methods=['POST'])
def sketch():
    mode = request.form.get('mode')
    image = None

    if 'image' in request.files and request.files['image'].filename != '':
        image = Image.open(request.files['image'].stream)

    if image:
        if mode == 'bw':
            result = create_bw_sketch(image)
        elif mode == 'color':
            result = create_color_sketch(image)
        elif mode == 'cartoon':
            result = create_cartoon_effect(image)
        elif mode == 'oil':
            result = create_oil_effect(image)
        else:
            result = image

        result.save(RESULT_PATH)
        return render_template('index.html', image_url='/' + RESULT_PATH)

    return render_template('index.html', image_url=None)

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_file(os.path.join(UPLOAD_FOLDER, filename), mimetype='image/jpeg')

if __name__ == '__main__':
    import os
port = int(os.environ.get("PORT", 5000))
app.run(debug=True, host="0.0.0.0", port=port)
