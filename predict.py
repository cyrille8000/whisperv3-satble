from cog import BasePredictor, Path, Input
import stable_whisper
import torch
import json

class Predictor(BasePredictor):
    def setup(self):
        """Setup method can be used for initial configurations if needed."""
        pass

    def predict(self,
                size: str = Input(description="Model size: 'base' or 'large'", default='base'),
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
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # Load the model based on the 'size' argument
        model_path = './base.pt' if size == 'base' else './large-v3.pt'
        self.model = stable_whisper.load_model(model_path, device=device)

        # Perform the transcription
        result = self.model.transcribe(str(audio_file), regroup=regroup, demucs=demucs, vad=vad, mel_first=mel_first)

        # Convert the result to a JSON string
        result_json = json.dumps(result, ensure_ascii=False)

        return result_json
