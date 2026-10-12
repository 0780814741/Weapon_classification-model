# Weapon Detection Model

An AI model that detects weapons in images and a Flask web app that serves it.
Built by **Tuyizere Christian**  Kigali.

**Live app:** https://weapon-classification-model.onrender.com

> The app runs on a free server. If it has been idle, the first load can take about a minute, and a scan can take 10 to 40 seconds.

## What it does
Upload an image and the app draws boxes around any weapons it finds, with a confidence score for each. It detects 4 classes:

1. Handgun
2. Knife
3. Rifle
4. Sword

## Model and dataset
- **Model:** YOLO11n (Ultralytics), a convolutional neural network (CNN) object detector, fine-tuned from pretrained weights
- **Dataset:** Roboflow *weapon-detection v4* (CC BY 4.0), 1,863 images with 2,103 labelled boxes
  https://universe.roboflow.com/maheshchhetri/weapon-detection-e6otc/dataset/4
- **Split:** 80% train, 10% validation, 10% test, stratified by class
- **Training:** up to 100 epochs with early stopping (patience 20), image size 640, batch size 16, on a Google Colab GPU

| Metric (test set) | Result |
|---|---|
| mAP50 | _add your value from the notebook_ |
| mAP50-95 | _add your value from the notebook_ |
| Precision | _add your value from the notebook_ |
| Recall | _add your value from the notebook_ |

## Project steps (done in the Colab notebook)
1. Dataset: load and display the data
2. Feature engineering: box area, aspect ratio, objects per image
3. Feature extraction: colour, edge and HOG features, shown with PCA
4. Split the dataset into train, validation and test
5. Train the model using a CNN (YOLO11)
6. Deploy with Flask on Render
7. Visualisation: training curves, confusion matrix, sample predictions

## Files in this repo
| File | Purpose |
|---|---|
| `app.py` | Flask app that loads the model and runs detection |
| `best.pt` | Trained model weights |
| `requirements.txt` | Python packages |
| `templates/index.html` | Web page |
| `.python-version` | Python version used on Render |

## Run it on your own computer
```bash
pip install -r requirements.txt
flask --app app run
```
Then open http://127.0.0.1:5000 in your browser.

## Deployment (Render)
- Build command: `pip install -r requirements.txt && pip install --force-reinstall --no-deps opencv-python-headless`
- Start command: `gunicorn app:app --workers 1 --timeout 120`

## Limitations
- It only recognises the 4 classes above. Other objects, such as an axe or a bat, may be missed or mislabelled.
- It is an automated screening tool. A clear result does not guarantee that no weapon is present.
- It is not a safety system and should not be used to make real security decisions.
