import stable_whisper
import json
import re
from flask import Flask


# Determine the device to use (GPU if available, otherwise CPU)
device = "cuda"

# Load the model based on the 'size' argument
model = stable_whisper.load_model("whisper-cache/large-v3.pt" , device=device)

def parse_srt_file(filepath):
    """Parse un fichier SRT et retourne les données dans un format structuré."""
    with open(filepath, 'r', encoding='utf-8') as file:
        content = file.read()
        
    # Regex pour trouver les temps et les textes
    pattern = re.compile(r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\s*(.*?)(?=\n\n|\Z)', re.DOTALL)
    matches = pattern.findall(content)

    segments = []
    for start, end, text in matches:
        segment = {
            "text": text.strip().replace('\n', ' '),
            "start": start,
            "end": end
        }
        segments.append(segment)

    return segments

@app.route('/')
def main():
    
    # Perform the transcription
    result = model.transcribe("segment2.wav", regroup='sp=.* /。/!/?/？+1', demucs=True, vad=True, mel_first=True)
    
    result.to_srt_vtt('./audio.srt', segment_level=True, word_level=False)
    
    srt_data = parse_srt_file('./audio.srt')
    
    # Convertir en JSON
    json_data = json.dumps({"segmentation": srt_data}, ensure_ascii=False, indent=4)

    return json_data

app = Flask(__name__)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
    
