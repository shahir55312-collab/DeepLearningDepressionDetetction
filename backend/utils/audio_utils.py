import moviepy.editor as mp

def extract_audio(video_path):
    clip = mp.VideoFileClip(video_path)
    audio_path = "temp_audio.wav"
    clip.audio.write_audiofile(audio_path)
    return audio_path