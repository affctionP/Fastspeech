        
import librosa

# def wav_to_mel_spectrogram(source_audio_path):
#     try :        
#         audio, _ = librosa.core.load(source_audio_path+'.wav', sr=config["mel_params"]["sample_rate"])
#         audio = trim_silence(audio, config["ref_level_db"])
#         melspec = log_melspectrogram(audio, **config["mel_params"])
#         np.save(out_mel_path, melspec)
#         meta_line = f"{file_name}|{speaker_name}|{text}|{phoneme}|{melspec.shape[1]}|{phoneme_idx}"
#         return meta_line

#     except Exception as e:
#         #import traceback
#         print(f"Error in processing {file_name} error {e}")
#         #traceback.print_exc()
#         return None




import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

def get_mel_spectrogram(wav_file_path, n_mels=128, hop_length=512, n_fft=2048):
    """
    Generate a mel spectrogram from a .wav file.
    Parameters:
        wav_file_path (str): Path to the .wav file.
        n_mels (int): Number of mel bands to generate.
        hop_length (int): Number of samples between successive frames.
        n_fft (int): Length of the FFT window.

    Returns:
        np.ndarray: Mel spectrogram.
    """
    # Load the audio file
    y, sr = librosa.load(wav_file_path, sr=None)
    
    # Generate the mel spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    
    # Convert to log scale (dB)
    log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    print(f"shape of {mel_spectrogram.shape}")
    plot_mel_spectrogram(log_mel_spectrogram,sr)
    return log_mel_spectrogram


def plot_mel_spectrogram(mel_spectrogram, sr):
    """
    Plot the mel spectrogram.

    Parameters:
        mel_spectrogram (np.ndarray): Mel spectrogram to plot.
        sr (int): Sampling rate of the audio.
        hop_length (int): Number of samples between successive frames.
    """
    plt.figure(figsize=(10, 4))
    img = librosa.display.specshow(mel_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(img, format='%+2.0f dB')  # Pass 'img' to plt.colorbar
    plt.title('Mel Spectrogram')
    plt.tight_layout()
    plt.show()




wav_file_path = "/home/hosseini/Downloads/LJSpeech-1.1/wavs/LJ001-0003.wav"
# Passing through arguments to the Mel filters

get_mel_spectrogram(wav_file_path)

