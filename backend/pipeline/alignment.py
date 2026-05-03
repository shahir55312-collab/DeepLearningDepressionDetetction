def align_modalities(audio_features, video_features, text_features):
    seq_len = min(
        len(audio_features),
        len(video_features),
        len(text_features)
    )
    audio_features = audio_features[:seq_len]
    video_features = video_features[:seq_len]
    text_features = text_features[:seq_len]
    return audio_features, video_features, text_features
