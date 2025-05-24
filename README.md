# 📸 Image Deblurring Web App

A web application that allows users to upload blurry images and enhance them using image restoration techniques.

## 🚀 Features

- Upload blurry images
- Preview original and deblurred images
- Process images using Unsharp Mask and Wiener Filter
- Download enhanced images
- Modern and responsive UI

## 🛠️ Technologies Used

- Python 3.11
- Flask
- OpenCV & NumPy
- TailwindCSS
- HTML5

## 🏗️ Installation

1. Clone the repository:
```bash
git clone <repository-url>
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

4. Run the application:
```bash
flask --app app run
```

5. Open your browser and navigate to `http://localhost:5000`

## 📝 Usage

1. Click the upload button to select a blurry image
2. Preview the original image
3. Choose a deblurring method (Unsharp Mask or Wiener Filter)
4. Click "Process" to enhance the image
5. Download the enhanced image

## 🔬 Deblurring Methods

### Unsharp Masking
A technique that enhances image sharpness by adding a blurred version of the image to the original.

### Wiener Filter
A method that removes noise and blur based on noise statistics.

## 📄 License

MIT License 