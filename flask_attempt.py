import os
import cv2
import numpy as np
from flask import Flask, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return 'No file part'
        file = request.files['file']
        if file.filename == '':
            return 'No selected file'
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            results = analyze_image(filepath)
            return render_template('results.html', results=results, filename=filename)
    return render_template('upload.html')

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

def analyze_image(filepath):
    # Load the masked image
    masked_image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)

    # Perform connected component analysis
    num_labels, labels = cv2.connectedComponents(masked_image)

    results = []
    for label in range(1, num_labels):  # Start from 1 to skip background
        mask = labels == label
        pixel_count = np.sum(mask)

        # Get mask dimensions
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])

        results.append({
            'label': label,
            'pixel_count': pixel_count,
            'width': w,
            'height': h
        })

    return results

if __name__ == '__main__':
    app.run(debug=True)