import requests

url = "http://localhost:5000/predictions"

# Création du dictionnaire des paramètres
params = {
    "input": {
        "vad": True,
        "demucs": True,
        "regroup": "sp=.* /。/!/?/？+1",
        "mel_first": True,
        "audio_file": 'segment2.wav',
        "model_path": "large-v3",
        "word_level": False,
        "model_device": "cuda",
        "segment_level": True
    }
}

# Ajouter le fichier en tant que partie multipart/form-data
files = {
    'audio_file': open('segment2.wav', 'rb')
}

# Envoyer la requête POST avec les paramètres et le fichier
response = requests.post(url, json=params)
print(response.text)
