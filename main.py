import stable_whisper
model = stable_whisper.load_model('base')
result = model.transcribe('segment6.wav', regroup=True)
result.to_srt_vtt('audio.srt')
