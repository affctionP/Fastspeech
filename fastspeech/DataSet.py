import torch
from torch.utils.data import Dataset, DataLoader
from phonemizing import text_to_phonemes
from melgenerate import get_mel_spectrogram

def normalize_text(text):
        import re
        cleaned_text = re.sub(r'[\'"]', '', text)
        return cleaned_text
class LJSpeechDataset(Dataset):
    def __init__(self, metadata_path, wav_dir, phoneme_vocab):
        self.metadata = self.load_metadata(metadata_path)
        self.wav_dir = wav_dir
        self.phoneme_vocab = phoneme_vocab

    def load_metadata(self, path):
        with open(path, "r") as f:
            lines = f.readlines()
        return [line.strip().split("|") for line in lines]




    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        text = self.metadata[idx][1]  # Text transcript
        wav_path = f"{self.wav_dir}/{self.metadata[idx][0]}.wav"
        
        # Convert text to phonemes
        phonemes = text_to_phonemes(normalize_text(text))

        phonemes = torch.tensor(phonemes, dtype=torch.long)
        
        # Extract mel-spectrogram
        mel_spectrogram = get_mel_spectrogram(wav_path)
        mel_spectrogram = torch.tensor(mel_spectrogram, dtype=torch.float32)
        return phonemes, mel_spectrogram


metadata_path ="/home/hosseini/Downloads/LJSpeech-1.1/metadata.csv"
wave_dir="/home/hosseini/Downloads/LJSpeech-1.1/wavs"

# dataset=LJSpeechDataset(metadata_path,wave_dir,None)
# dataset[16]