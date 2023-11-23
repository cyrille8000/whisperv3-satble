import stable_whisper
model = stable_whisper.load_model('large-v3')
result = model.transcribe('segment6.wav')
result.to_srt_vtt('audio.srt')
