import librosa
import noisereduce as nr
import soundfile as sf

def remove_noise(input_path, output_path):
    y, sr = librosa.load(input_path, sr=None)
    reduced_noise = nr.reduce_noise(y=y, sr=sr)
    sf.write(output_path, reduced_noise, sr)
