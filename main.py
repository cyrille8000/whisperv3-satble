import requests
import base64

url = "http://localhost:5000/predictions"

# Chemin vers le fichier à envoyer
file_path = "segment2.wav"

# Lire et encoder le fichier en base64
with open(file_path, 'rb') as file:
    encoded_file = base64.b64encode(file.read()).decode('utf-8')

# Préparer les données JSON avec le fichier encodé
data = {
    "vad": True,
    "demucs": True,
    "regroup": "sp=.* /。/!/?/？+1",
    "mel_first": True,
    "audio_file": encoded_file,
    "model_path": "large-v3",
    "word_level": False,
    "model_device": "cuda",
    "segment_level": True
}

# Envoyer la requête POST avec les données JSON
response = requests.post(url, json={'input': data})
