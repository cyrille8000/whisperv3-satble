from cog import BasePredictor, Path, Input
import stable_whisper

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        self.model = stable_whisper.load_model('base')

    def predict(self,
                audio_file: Path = Input(description="Path to the audio file"),
                regroup: str = Input(description="Regrouping pattern for transcription", default='sp=.* /。/!/?/？+1'),
                demucs: bool = Input(description="Use Demucs for denoising", default=True),
                vad: bool = Input(description="Use VAD (Voice Activity Detection)", default=True),
                mel_first: bool = Input(description="Process mel spectrogram first", default=True),
                output_format: str = Input(description="Output format: 'srt' or 'vtt'", default='srt'),
                segment_level: bool = Input(description="Segment level transcription", default=True),
                word_level: bool = Input(description="Word level transcription", default=False)
                ) -> Path:
        """Run a single prediction on the model"""
        # Transcription
        result = self.model.transcribe(audio_file, regroup=regroup, demucs=demucs, vad=vad, mel_first=mel_first)

        # Output file path
        output_file = Path("transcription." + output_format)

        # Generating transcription file
        if output_format == 'srt':
            result.to_srt_vtt(str(output_file), segment_level=segment_level, word_level=word_level)
        elif output_format == 'vtt':
            result.to_srt_vtt(str(output_file), segment_level=segment_level, word_level=word_level)

        return output_file
