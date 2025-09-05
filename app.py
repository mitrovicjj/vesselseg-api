import torch
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from unet import UNet
from torchvision import transforms
from PIL import Image
import io
import numpy as np
import os
import tempfile
import requests

app = FastAPI()

MODEL_URL = "https://drive.google.com/uc?id=1bCoO6XMGwD_Qob9ljxRy94AWaolCkST7"
MODEL_PATH = "unet.pth"
DEVICE = torch.device("cpu")

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    r = requests.get(MODEL_URL, stream=True)
    with open(MODEL_PATH, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    print("Download complete.")

model = UNet(n_channels=3, n_classes=1)
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)


transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

@app.get("/")
def home():
    return {"message": "UNet API is running!"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    orig_size = image.size
    image_resized = image.resize((256, 256))

    # Koristi float16 tensor
    tensor = transform(image).unsqueeze(0).half()  # (1,3,256,256)

    # inference
    with torch.no_grad():
        output = model(tensor)
        pred_mask = torch.sigmoid(output).squeeze().cpu().numpy()

    mask_bin = (pred_mask > 0.5).astype(np.uint8) * 255
    mask_img = Image.fromarray(mask_bin).resize(orig_size)

    # overlay
    overlay = image.convert("RGBA")
    mask_rgba = Image.new("RGBA", overlay.size, (255, 0, 0, 0))
    mask_pixels = mask_img.load()
    overlay_pixels = mask_rgba.load()

    for y in range(mask_img.height):
        for x in range(mask_img.width):
            if mask_pixels[x, y] > 0:
                overlay_pixels[x, y] = (255, 0, 0, 100)

    final_overlay = Image.alpha_composite(overlay, mask_rgba)

    tmp_dir = tempfile.gettempdir()
    out_path = os.path.join(tmp_dir, "prediction_overlay.png")
    final_overlay.save(out_path, format="PNG")

    return FileResponse(out_path, media_type="image/png", filename="prediction_overlay.png")
