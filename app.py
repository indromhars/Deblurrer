import streamlit as st
import numpy as np
from PIL import Image
import io

# Try to import OpenCV, if not available, use PIL for basic operations
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    st.warning("OpenCV is not available. Some advanced image processing features may be limited.")

def unsharp_mask(image, kernel_size=(9, 9), sigma=2.0, amount=4.0, threshold=0):
    """Apply unsharp mask to the image with more aggressive parameters."""
    if not OPENCV_AVAILABLE:
        # Fallback to PIL-based implementation
        img_array = np.array(image)
        blurred = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=sigma))
        blurred_array = np.array(blurred)
        diff = img_array - blurred_array
        mask = np.abs(diff) > threshold
        result = img_array + amount * diff * mask
        result = np.clip(result, 0, 255).astype(np.uint8)
        return result
    
    # OpenCV implementation
    image = image.astype(np.float32)
    blurred = cv2.GaussianBlur(image, kernel_size, sigma)
    diff = image - blurred
    mask = np.abs(diff) > threshold
    result = image + amount * diff * mask
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

def wiener_filter(image, kernel_size=(5, 5), K=0.001):
    """Apply Wiener filter to the image with improved implementation."""
    if not OPENCV_AVAILABLE:
        st.error("Wiener filter is not available without OpenCV. Please install OpenCV for full functionality.")
        return image
    
    # Convert to float32 for better precision
    image = image.astype(np.float32)
    
    # Process each channel separately
    result = np.zeros_like(image)
    for i in range(3):  # Process each color channel
        # Create Gaussian blur kernel
        kernel = cv2.getGaussianKernel(kernel_size[0], 0)
        kernel = kernel * kernel.T
        
        # Pad the image to avoid edge effects
        pad_size = kernel_size[0] // 2
        padded = cv2.copyMakeBorder(image[:,:,i], pad_size, pad_size, pad_size, pad_size, cv2.BORDER_REFLECT)
        
        # Apply FFT
        image_fft = np.fft.fft2(padded)
        kernel_fft = np.fft.fft2(kernel, s=padded.shape)
        
        # Wiener filter with improved noise reduction
        kernel_fft_conj = np.conj(kernel_fft)
        wiener = kernel_fft_conj / (np.abs(kernel_fft) ** 2 + K)
        
        # Apply filter
        result_fft = image_fft * wiener
        result_padded = np.real(np.fft.ifft2(result_fft))
        
        # Remove padding
        result[:,:,i] = result_padded[pad_size:-pad_size, pad_size:-pad_size]
    
    # Normalize and convert back to uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    return result

# Set page config
st.set_page_config(
    page_title="Image Deblurrer",
    page_icon="🖼️",
    layout="wide"
)

# Title and description
st.title("Image Deblurrer")
st.write("Upload an image and choose a deblurring method to enhance its clarity.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=['png', 'jpg', 'jpeg'])

if uploaded_file is not None:
    # Read image
    image = Image.open(uploaded_file)
    image_array = np.array(image)
    
    # Convert to RGB if needed
    if len(image_array.shape) == 2:  # Grayscale
        image_array = np.stack((image_array,) * 3, axis=-1)
    elif image_array.shape[2] == 4:  # RGBA
        image_array = image_array[:, :, :3]
    
    # Create two columns for original and processed images
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_column_width=True)
    
    # Processing method selection
    method = st.radio("Select deblurring method:", ["Unsharp Mask", "Wiener Filter"])
    
    # Process button
    if st.button("Process Image"):
        with st.spinner("Processing image..."):
            # Process the image
            if method == "Unsharp Mask":
                processed = unsharp_mask(image_array)
            else:  # wiener
                processed = wiener_filter(image_array)
            
            # Convert to PIL Image for display
            processed_image = Image.fromarray(processed)
            
            with col2:
                st.subheader("Processed Image")
                st.image(processed_image, use_column_width=True)
                
                # Add download button
                buf = io.BytesIO()
                processed_image.save(buf, format="PNG")
                st.download_button(
                    label="Download processed image",
                    data=buf.getvalue(),
                    file_name="deblurred.png",
                    mime="image/png"
                )