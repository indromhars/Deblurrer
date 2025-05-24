import os
from flask import Flask, render_template, request, jsonify, send_file
import cv2
import numpy as np
from werkzeug.utils import secure_filename
from PIL import Image
import io

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def unsharp_mask(image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
    """Apply unsharp mask to the image."""
    # Convert to float32 for better precision
    image = image.astype(np.float32)
    
    # Create Gaussian blur
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    
    # Calculate the difference
    diff = image - blurred
    
    # Apply threshold
    mask = np.abs(diff) > threshold
    
    # Apply the mask
    result = image + amount * diff * mask
    
    # Clip values to valid range
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

def wiener_filter(image, kernel_size=(5, 5), K=0.01):
    """Apply Wiener filter to the image."""
    # Convert to float32 for better precision
    image = image.astype(np.float32)
    
    # Create Gaussian blur kernel
    kernel = cv2.getGaussianKernel(kernel_size[0], 0)
    kernel = kernel * kernel.T
    
    # Apply FFT
    image_fft = np.fft.fft2(image)
    kernel_fft = np.fft.fft2(kernel, s=image.shape)
    
    # Wiener filter
    kernel_fft_conj = np.conj(kernel_fft)
    wiener = kernel_fft_conj / (np.abs(kernel_fft) ** 2 + K)
    
    # Apply filter
    result_fft = image_fft * wiener
    result = np.real(np.fft.ifft2(result_fft))
    
    # Normalize and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    return result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file and allowed_file(file.filename):
        # Read image
        img_stream = file.read()
        nparr = np.frombuffer(img_stream, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Get processing method
        method = request.form.get('method', 'unsharp')
        
        # Process image
        if method == 'unsharp':
            processed = unsharp_mask(img)
        else:  # wiener
            processed = wiener_filter(img)
        
        # Convert to bytes
        _, buffer = cv2.imencode('.png', processed)
        img_bytes = buffer.tobytes()
        
        return send_file(
            io.BytesIO(img_bytes),
            mimetype='image/png',
            as_attachment=True,
            download_name='deblurred.png'
        )
    
    return jsonify({'error': 'Invalid file type'}), 400

if __name__ == '__main__':
    app.run(debug=True) 