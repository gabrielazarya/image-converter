from flask import Flask, render_template, request, send_file
import cv2
import numpy as np
from io import BytesIO
import base64

app = Flask(__name__)

def convert_to_cartoon(image_path):
    img = cv2.imread(image_path)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)

    color = cv2.bilateralFilter(img, 9, 250, 250)
    cartoon = cv2.bitwise_and(color, color, mask=edges)

    return cartoon

def remove_background(image_path):
    img = cv2.imread(image_path)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    rect = (50, 50, img.shape[1]-50, img.shape[0]-50)
    cv2.grabCut(img, mask, rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    img = img * mask2[:, :, np.newaxis]

    return img

def create_anaglyph(image_left_path, image_right_path):
    img_left = cv2.imread(image_left_path)
    img_right = cv2.imread(image_right_path)

    # Pastikan ukuran kedua gambar sama
    img_right = cv2.resize(img_right, (img_left.shape[1], img_left.shape[0]))

    # Pisahkan saluran warna dari masing-masing gambar
    b_left, g_left, r_left = cv2.split(img_left)
    b_right, g_right, r_right = cv2.split(img_right)

    # Gabungkan saluran warna untuk membuat gambar anaglyph
    anaglyph = cv2.merge((b_left, g_right, r_right))

    return anaglyph

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/convert_cartoon', methods=['GET', 'POST'])
def convert_cartoon():
    return render_template('convert_cartoon.html')

@app.route('/filter_anaglyph', methods=['GET', 'POST'])
def anaglyph():
    return render_template('filter_anaglyph.html')

@app.route('/remove_background', methods=['GET', 'POST'])
def remove_bg():
    return render_template('remove_background.html')

@app.route('/upload_cartoon', methods=['POST'])
def upload1():
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        image_path = 'uploads/cartoon/input_photo.jpg'
        cartoon_path = 'download/cartoon/output_cartoon.jpg'

        # Convert to Cartoon
        photo.save(image_path)
        cartoon = convert_to_cartoon(image_path)
        cv2.imwrite(cartoon_path, cartoon)

        return render_template('convert_cartoon.html', cartoon_path=cartoon_path)

    return render_template('convert_cartoon.html', error='Upload failed. Please try again.')

@app.route('/upload_no_bg', methods=['POST'])
def upload2():
    if request.method == 'POST' and 'photo' in request.files:
        photo = request.files['photo']
        image_path = 'uploads/no_bg/input_photo.jpg'
        no_bg_path = 'download/no_bg/output_no_bg.jpg'
        
        # Remove BG
        photo.save(image_path)
        no_bg_image = remove_background(image_path)
        cv2.imwrite(no_bg_path, no_bg_image)

        return render_template('remove_background.html', no_bg_path=no_bg_path)

    return render_template('remove_background.html', error='Upload failed. Please try again.')

@app.route('/upload_anaglyph', methods=['POST'])
def filter_anaglyph():
    if request.method == 'POST' and 'left_photo' in request.files and 'right_photo' in request.files:
        left_photo = request.files['left_photo']
        right_photo = request.files['right_photo']

        left_path = 'uploads/anaglyph/left_photo.jpg'
        right_path = 'uploads/anaglyph/right_photo.jpg'
        anaglyph_path = 'download/anaglyph/output_anaglyph.jpg'

        left_photo.save(left_path)
        right_photo.save(right_path)

        anaglyph_image = create_anaglyph(left_path, right_path)
        cv2.imwrite(anaglyph_path, anaglyph_image)

        return render_template('filter_anaglyph.html', anaglyph_path=anaglyph_path)

    return render_template('filter_anaglyph.html', error='Upload failed. Please try again.')

@app.route('/download_cartoon')
def download_cartoon():
    cartoon_path = 'download/cartoon/output_cartoon.jpg'
    return send_file(cartoon_path, as_attachment=True)

@app.route('/download_no_bg')
def download_no_bg():
    no_bg_path = 'download/no_bg/output_no_bg.jpg'
    return send_file(no_bg_path, as_attachment=True)

@app.route('/download_anaglyph')
def download_anaglyph():
    anaglyph_path = 'download/anaglyph/output_anaglyph.jpg'
    return send_file(anaglyph_path, as_attachment=True)
    
if __name__ == '__main__':
    app.run(debug=True)