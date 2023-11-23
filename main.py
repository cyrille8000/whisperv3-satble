import stable_whisper
model = stable_whisper.load_model('base')
result = model.transcribe('episode3.wav', regroup='cm_sp=.* / cm_sp=?* / cm_sp=!*', demucs=True, vad=True, mel_first=True)
result.to_srt_vtt('audio.srt', segment_level=True, word_level=False)
