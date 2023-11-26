import requests
import json

url = "http://localhost:5000/predictions"

# Chemin vers le fichier à envoyer
file_path = "segment2.wav"

# Paramètres supplémentaires à envoyer avec le fichier
params = {
    "vad": True,
    "demucs": True,
    "regroup": "sp=.* /。/!/?/？+1",
    "mel_first": True,
    "model_path": "large-v3",
    "word_level": False,
    "model_device": "cuda",
    "segment_level": True
}

# Préparer le corps de la requête
data = {'input': json.dumps(params)}

# Ouvrir le fichier en mode binaire
with open(file_path, 'rb') as file:
    # Créer un dictionnaire pour les fichiers
    files = {'audio_file': (file_path, file)}

    # Envoyer la requête POST avec les données et le fichier
    response = requests.post(url, data=data, files=files)

    # Afficher la réponse du serveur
    print(response.text)
