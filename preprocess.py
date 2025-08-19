from pydub import AudioSegment, effects


def preprocess_audio(input_file):
    file= open(input_file, "rb")
    audio = AudioSegment.from_file(file)
    new_audio = audio.set_frame_rate(16000) 
    new_audio = effects.normalize(new_audio)  # Normalize the audio
    new_audio = new_audio.set_channels(1)  # Convert to mono   
    new_audio.export("processed.mp3", format="mp3", bitrate="32k")


preprocess_audio("test.mp4")