import stable_whisper
model = stable_whisper.load_model('base')
result = model.transcribe('segment2.wav', regroup='sp=.* /。/!/?/？+1', demucs=True, vad=True, mel_first=True)
result.to_srt_vtt('audio.srt', segment_level=True, word_level=False)
