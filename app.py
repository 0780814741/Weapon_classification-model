import base64
import cv2
import numpy as np
from flask import Flask, render_template, request
from ultralytics import YOLO

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024        # 8 MB upload limit
model = YOLO('best.pt')
model(np.zeros((640, 640, 3), np.uint8), verbose=False)   # warm-up so the first real request is not the slowest

def to_b64(img):
    return base64.b64encode(cv2.imencode('.jpg', img)[1]).decode()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    file = request.files.get('file')
    img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR) if file else None
    if img is None:
        return render_template('index.html', error='Please upload a valid image.')
    result = model(img, conf=0.25)[0]
    detections = [(model.names[int(b.cls)], round(float(b.conf), 2)) for b in result.boxes]
    return render_template('index.html', original=to_b64(img), result=to_b64(result.plot()), detections=detections)
