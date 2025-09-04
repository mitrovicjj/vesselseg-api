# UNet Segmentation API

This project implements a FastAPI application for semantic segmentation using a UNet model trained for binary segmentation. After uploading an image, the API returns the result with an overlay prediction mask.

The implementation includes several modifications to the standard UNet architecture to improve performance and reduce memory usage.

---

## Project Overview

The project consists of:
- FastAPI backend for the REST API
- Custom UNet model implemented in PyTorch (`unet.py`)
- Image preprocessing and postprocessing pipeline
- Binary segmentation with overlay visualization
- Optimized for CPU inference

---

## Architecture

### Model architecture (`unet.py`)

The UNet implementation follows the encoder-decoder structure with skip connections:

- **DoubleConv blocks**: Two convolution layers with normalization and activation
- **Encoder path**: Downsampling using MaxPool and feature extraction
- **Decoder path**: Upsampling, skip connections from the encoder
- **Skip connections**: Combine encoder and decoder features for better localization

### API implementation (`app.py`)

The FastAPI application handles:
- **Model loading**: Loads pre-trained weights from Google Drive
- **Image processing**: Handles resizing, normalization, and tensor conversion
- **Inference**: Forward pass on CPU (float16 to reduce memory usage)
- **Postprocessing**: Sigmoid activation and thresholding, creates overlay visualization

---

## Application

1. **Image upload**: Upload image via POST request to /predict
2. **Preprocessing**:
   - Resize image to 256×256 pixels
   - Normalize and convert to tensor
   - Use half-precision for efficiency
3. **Model inference**:
   - Forward pass through UNet
   - Sigmoid activation, threshold at 0.5
4. **Postprocessing**:
   - Create semi-transparent overlay
   - Combine with original image
5. **Response**: Return processed image as PNG

---

## API Endpoints

The API provides two endpoints:

GET / – A health check endpoint that returns a JSON message:
{"message": "UNet API is running!"}

POST /predict – Accepts image file and returns the segmentation result with the overlay applied.

---

## Installation steps

1. **Clone the repository**
```bash
git clone <repository-url>
cd unet-segmentation-api
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate     # Windows
```

3. **Install dependencies**
```bash
pip install fastapi uvicorn torch torchvision pillow requests
```

4. **Run the application**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

5. **Access the API**
   - API documentation: `http://localhost:8000/docs`
   - Health check: `http://localhost:8000/`

---

## UNet implementation details

### Modifications

#### 1. Used InstanceNorm because it works better for small batches
```python
nn.InstanceNorm2d(out_channels, affine=True)
```


#### 2. Added dropout to prevent overfitting
```python
self.up1 = Up(512+256, 256, dropout=0.3)
self.up2 = Up(256+128, 128, dropout=0.2) 
self.up3 = Up(128+64, 64, dropout=0.1)
self.up4 = Up(64+32, 32, dropout=0.0)
```
- Gradual dropout reduction
- Applied only in upsampling blocks for regularization

#### 3. Reduced channel dimensions
- Encoder: 32 → 64 → 128 → 256 → 512 (instead of starting at 64)
- Reduces memory usage while maintaining performance
- Suitable for less complex segmentation tasks

---

## Attempted Vercel deployment

Tried deploying to Vercel but ran into issues:

- Memory limit (~512MB on serverless functions)

- PyTorch size exceeds available resources

For now, the API runs locally or on a server with enough memory.
---

## Usage examples

### Python Client example
```python
import requests

url = "http://localhost:8000/predict"
files = {"file": open("image.jpg", "rb")}
response = requests.post(url, files=files)

with open("result.png", "wb") as f:
    f.write(response.content)
```

Or use Swagger UI at: http://localhost:8000/docs

---

## Future improvements

- Add GPU support
- Enable batch processing
- Experiment with model ensembles
