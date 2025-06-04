# 📸 Deblurrer
A web application that allows users to upload blurry images and enhance them using image restoration techniques.

## 🚀 Features

- Upload blurry images
- Preview original and deblurred images
- Process images using Unsharp Mask and Wiener Filter
- Download enhanced images
- Modern and responsive UI
- Quality metrics: MSE & PSNR

## 🛠️ Technologies Used

- Python 3.11
- Streamlit
- OpenCV & NumPy
- Pillow

## 🏗️ Installation

1. Clone the repository:
```bash
git clone https://github.com/indromhars/Deblurrer.git
cd deblurrer
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. (For deployment) Install system packages:
If deploying on Streamlit Cloud, make sure `packages.txt` is present with:
```
libgl1-mesa-glx
libglib2.0-0
```

5. Run the application:
```bash
streamlit run app.py
```

6. Open your browser and navigate to `http://localhost:8501`

## 📝 Usage

1. Click the upload button to select a blurry image
2. Preview the original image
3. Choose a deblurring method (Unsharp Mask or Wiener Filter)
4. Click "Process" to enhance the image
5. View MSE and PSNR metrics for quality assessment
6. Download the enhanced image

## 🔬 Deblurring Methods & Metrics

### Unsharp Masking
A technique that enhances image sharpness by adding a blurred version of the image to the original. Tuned parameters for best results: `kernel_size=(5,5)`, `sigma=1.0`, `amount=1.5`.

### Wiener Filter
A method that removes noise and blur based on noise statistics. Tuned parameters for best results: `kernel_size=(3,3)`, `K=0.01`.

### MSE (Mean Squared Error)
Measures the average squared difference between the original and processed image pixels. Lower values mean better quality.

### PSNR (Peak Signal-to-Noise Ratio)
Expresses the ratio between the maximum possible power of a signal and the power of corrupting noise. Higher values mean better quality.

## ⚙️ Parameter Tuning
You can further tune the parameters in `app.py` for your specific images. Try adjusting `kernel_size`, `sigma`, `amount`, and `K` for best results.

## 📄 License

MIT License 