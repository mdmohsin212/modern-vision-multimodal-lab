from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter
import numpy as np
import requests

BASE = "http://127.0.0.1:8000"
IMAGE = Path("samples/road.jpg").read_bytes()
N = 40


def send(_):
    started = perf_counter()

    response = requests.post(
        f"{BASE}/predict",
        params={"conf": 0.25},
        files={"file": ("road.jpg", IMAGE, "image/jpeg")},
        timeout=120,
    )

    client_ms = (perf_counter() - started) * 1000
    response.raise_for_status()
    result = response.json()

    return result["server_processing_ms"], client_ms

health = requests.get(f"{BASE}/health", timeout=10)
health.raise_for_status()
print(health.json())

for _ in range(3):
    send(0)

for concurrency in (1, 4):
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        started = perf_counter()
        rows = list(pool.map(send, range(N)))
        total_seconds = perf_counter() - started

    times = np.asarray(rows)

    print(
        f"concurrency={concurrency} | "
        f"server_mean={times[:, 0].mean():.2f} ms | "
        f"client_mean={times[:, 1].mean():.2f} ms | "
        f"client_p95={np.percentile(times[:, 1], 95):.2f} ms | "
        f"throughput={N / total_seconds:.2f} requests/s"
    )