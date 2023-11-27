from flask import Flask, request, jsonify
import stable_whisper
import json
import re

app = Flask(__name__)

# Initialisez le modèle ici
device = "cuda"
model = stable_whisper.load_model("whisper-cache/large-v3.pt", device=device)

def parse_srt_file(srt_content):
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s*(.*?)(?=\n\n|\Z)', re.DOTALL)
    matches = pattern.findall(srt_content)
    segments = [{"text": text.strip().replace('\n', ' '), "start": start, "end": end} for start, end, text in matches]
    return segments

@app.route('/transcribe', methods=['POST'])
def transcribe():
    # Obtenez le fichier audio de la requête
    audio_file = request.files['audio']

    # Transcription
    result = model.transcribe(audio_file, regroup='sp=.* /。/!/?/？+1', demucs=True, vad=True, mel_first=True)
    srt_content = result.to_srt_vtt(segment_level=True, word_level=False)

    # Parsez et retournez le JSON
    segments = parse_srt_file(srt_content)
    return jsonify({"segmentation": segments})

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)
