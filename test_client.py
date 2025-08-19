import requests
import base64

# lokalno  testiranje
url = "http://127.0.0.1:8000/predict"
file_path = "test.jpg"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status:", response.status_code)
data = response.json()
print("Response JSON:", {k: v for k, v in data.items() if k != "mask_base64"})

if "mask_base64" in data:
    mask_base64 = data["mask_base64"]
    mask_bytes = base64.b64decode(mask_base64)

    output_path = "predicted_mask.png"
    with open(output_path, "wb") as out_file:
        out_file.write(mask_bytes)

    print(f"Mask saved as {output_path}")