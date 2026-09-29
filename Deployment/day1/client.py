from pathlib import Path
from time import perf_counter
import requests

BASE = "http://127.0.0.1:8000"
IMAGE_PATH = Path(__file__).resolve().parent / "samples" / "road.jpg"


for number in range(1, 4):
    with IMAGE_PATH.open("rb") as image:
        started = perf_counter()
        
        response = requests.post(
            f"{BASE}/predict",
            params={"conf": 0.25},
            files={"file": (IMAGE_PATH.name, image)},
            timeout=120
        )
        
        client_ms = (perf_counter() - started) * 1000
    
    response.raise_for_status()
    result = response.json()
    
    print(
        f"Request {number}: "
        f"detections={result['num_detections']}, "
        f"server={result['server_processing_ms']:.2f} ms, "
        f"client={client_ms:.2f} ms"
    )


response = requests.post(
    f"{BASE}/predict",
    files={
        "file": ("fake.jpg", b"This is not an image", "image/jpeg")
    },
    timeout=30
)
print("Invalid file:", response.status_code, response.json())


with IMAGE_PATH.open("rb") as image:
    response = requests.post(
        f"{BASE}/predict",
        params={"conf": 1.0},
        files={"file": (IMAGE_PATH.name, image)},
        timeout=120,
    )

response.raise_for_status()
result = response.json()
print("Empty detections:", response.status_code, result["detections"])