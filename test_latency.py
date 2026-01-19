import requests
import time

url = "https://jayant--chatterbox-tts-ttsservice-api.modal.run/stream"
params = {
    "text": "Latency test.",
    "language": "en",
    "voice": "anaya_en_female"
}

print(f"Testing TTFB for: {url}")
start = time.perf_counter()

with requests.post(url, params=params, stream=True) as r:
    r.raise_for_status()
    # Read the first byte to measure TTFB
    first_byte = next(r.iter_content(chunk_size=1))
    ttfb = (time.perf_counter() - start) * 1000
    print(f"✅ Time To First Byte (TTFB): {ttfb:.2f}ms")
    
    # Read the rest to see total time
    for chunk in r.iter_content(chunk_size=4096):
        pass
    
    total = (time.perf_counter() - start) * 1000
    print(f"✅ Total Stream Finished: {total:.2f}ms")
