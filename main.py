import stable_whisper
model = stable_whisper.load_model('base')
result = model.transcribe('segment6.wav', regroup='cm_sp=.* /。/?/？/,* /，_sg=.5_mg=.3+3_sp=.* /。/?/？',segment_level=True, word_level=False)
result.to_srt_vtt('audio.srt')
