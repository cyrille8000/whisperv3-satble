import os
import whisper

print("ouii 1")
os.makedirs("whisper-cache", exist_ok=True)

print("ouii 2")
models = whisper.available_models()

print("ouii 3")
for model in models:
    print(f"Downloading {model}...")
    whisper._download(whisper._MODELS[model], "whisper-cache", in_memory=False)
