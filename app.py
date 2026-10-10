import os, time, uuid, glob
import cv2
from flask import Flask, render_template, request, url_for
from ultralytics import YOLO

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 8 * 1024 * 1024          # 8 MB upload limit
UPLOAD, RESULT = [os.path.join(BASE, 'static', d) for d in ('uploads', 'results')]
ALLOWED = {'png', 'jpg', 'jpeg', 'webp'}
CONF = float(os.environ.get('CONF_THRESHOLD', 0.25))
for d in (UPLOAD, RESULT):
    os.makedirs(d, exist_ok=True)
model = YOLO(os.path.join(BASE, 'best.pt'))

def prune(max_age=900):                                      # delete files older than 15 minutes
    for f in glob.glob(UPLOAD + '/*') + glob.glob(RESULT + '/*'):
        if time.time() - os.path.getmtime(f) > max_age:
            os.remove(f)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'GET':
        return render_template('index.html')
    prune()
    file = request.files.get('file')
    if not file or not file.filename:
        return render_template('index.html', error='Please choose an image.')
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in ALLOWED:
        return render_template('index.html', error='Unsupported file type. Use PNG, JPG or WEBP.')
    name = uuid.uuid4().hex + '.' + ext
    path = os.path.join(UPLOAD, name)
    file.save(path)
    try:
        r = model(path, conf=CONF)[0]
    except Exception:
        os.remove(path)
        return render_template('index.html', error='Could not read that image.')
    cv2.imwrite(os.path.join(RESULT, name), r.plot())
    detections = [(model.names[int(b.cls)], round(float(b.conf), 2)) for b in r.boxes]
    return render_template('index.html',
                           uploaded_image=url_for('static', filename='uploads/' + name),
                           result_image=url_for('static', filename='results/' + name),
                           detections=detections)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
