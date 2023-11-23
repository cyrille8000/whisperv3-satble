from cog import BasePredictor, Path, Input
import stable_whisper
import torch
import json
import re

class Predictor(BasePredictor):
    def setup(self):
        """Setup method can be used for initial configurations if needed."""
        pass

    def predict(self,
                size: str = Input(description="Model size: 'base' or 'large'", default='base'),
                model_device: str = Input(description="Model device: 'cpu' or 'cuda'", default='cpu'),
                audio_file: Path = Input(description="Path to the audio file"),
                regroup: str = Input(description="Regrouping pattern for transcription", default='sp=.* /。/!/?/？+1'),
                demucs: bool = Input(description="Use Demucs for denoising", default=True),
                vad: bool = Input(description="Use VAD (Voice Activity Detection)", default=True),
                mel_first: bool = Input(description="Process mel spectrogram first", default=True),
                segment_level: bool = Input(description="Segment level transcription", default=True),
                word_level: bool = Input(description="Word level transcription", default=False)
                ) -> str:
        """Run a single prediction on the model and return results as a JSON string"""

        # Determine the device to use (GPU if available, otherwise CPU)
        device = model_device

        # Load the model based on the 'size' argument
        model_path = './base.pt' if size == 'base' else './large-v3.pt'
        self.model = stable_whisper.load_model(model_path, device=device)

        # Perform the transcription
        result = self.model.transcribe(str(audio_file), regroup=regroup, demucs=demucs, vad=vad, mel_first=mel_first)

        result.to_srt_vtt('./audio.srt')
        
        srt_data = parse_srt_file('./audio.srt')
        
        # Convertir en JSON
        json_data = json.dumps({"segmentation": srt_data}, ensure_ascii=False, indent=4)


        return json_data

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
