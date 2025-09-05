import requests

# Local testing
url = "http://127.0.0.1:8000/predict"
file_path = "test.jpg"

with open(file_path, "rb") as f:
    files = {"file": f}
    response = requests.post(url, files=files)

print("Status:", response.status_code)

if response.status_code == 200:
    # Save the returned PNG file directly
    output_path = "predicted_mask.png"
    with open(output_path, "wb") as out_file:
        out_file.write(response.content)
    print(f"Prediction overlay saved as {output_path}")
else:
    print("Error:", response.text)
