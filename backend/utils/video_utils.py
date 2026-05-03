import cv2
import moviepy.editor as mp
import whisper


def load_video(video_path):
    return cv2.VideoCapture(video_path)


def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames=[]

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames


def resize_frame(frame):
    return cv2.resize(frame,(224,224))


def extract_audio(video_path):
    clip = mp.VideoFileClip(video_path)
    audio_path="temp_audio.wav"
    clip.audio.write_audiofile(audio_path)
    return audio_path


def transcribe_video(video_path):
    audio_path = extract_audio(video_path)

    model=whisper.load_model("base")
    result=model.transcribe(audio_path)

    return result["text"]