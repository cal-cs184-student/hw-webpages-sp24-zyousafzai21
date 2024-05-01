import librosa 
import numpy as np
import matplotlib.pyplot as plt


audio_file = "yeah30seconds.mp3"
audio_signal, sample_rate = librosa.load(audio_file, sr=None)

window_size = 2048
hop_length = 512

tempo, beat_frames = librosa.beat.beat_track(y=audio_signal, sr=sample_rate)

print("Estimated BPM:", tempo)

beat_frames_by_time = librosa.frames_to_time(beat_frames, sr = sample_rate)
print("Beat Frames By Time: ", beat_frames_by_time)

onset_env = librosa.onset.onset_strength(y=audio_signal, sr=sample_rate)
#print("Onset Strength:", onset_env)


spectrogram = librosa.stft(audio_signal, n_fft=window_size, hop_length=hop_length)
spectrogram = np.abs(spectrogram)
bass_spectrum = spectrogram[0:100, :]
sphere_speed = np.mean(bass_spectrum)
print("Sphere Speed: ", sphere_speed)